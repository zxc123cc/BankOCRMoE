import copy

import torch
import torch.nn as nn
from pathlib import Path
import time
import numpy as np
import random

import os
import torch.distributed as dist
import logging

from torch.cuda.amp import autocast
from transformers import get_scheduler, AdamW


def prepare_optimizer(model, opt_lr, weight_decay, eps):
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())
    optimizer_grouped_parameters = \
        [{'params': [p for n, p in model_param if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
         {'params': [p for n, p in model_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt_lr, eps=eps)

    return optimizer


def prepare_optimizer_delamination(model, igm_lr, opt_lr, weight_decay, eps):
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())

    igm_param_optimizer = []
    other_param_optimizer = []

    for name, param in model_param:
        if 'igm' in str(name):
            igm_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': opt_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt_lr},

        {"params": [p for n, p in igm_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': igm_lr},
        {"params": [p for n, p in igm_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': igm_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt_lr, eps=eps)

    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate,gradient_accumulation_steps=1):
    total_steps = (epochs * steps_per_epoch) / gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(args):
    time_ = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    log_file = os.path.join(args.save_model_path, f'{time_}.txt')
    logging.basicConfig(filename=log_file, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    return logger


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    print("dist", args)
    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    print(4)
    # set cuda device
    torch.cuda.set_device(args.gpu)


class Logger():
    def __init__(self, file_name, mode='w', buffer=100):
        (Path(file_name).parent).mkdir(exist_ok=True, parents=True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer

    def log(self, *args, end='\n'):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y) + ':' + str(x[y]) + ' ')
            else:
                self.fp.write(str(x) + ' ')
        self.fp.write(end)
        self.cnt += 1
        if self.cnt >= self.buffer or time.time() - self.stamp > 5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, 'a')
        pass

    def close(self):
        self.fp.close()

class EMA(object):
    """
    Maintains (exponential) moving average of a set of parameters.
    使用ema累积模型参数
    Args:
        parameters (:obj:`list`): 需要训练的模型参数
        decay (:obj:`float`): 指数衰减率
        use_num_updates (:obj:`bool`, optional, defaults to True): Whether to use number of updates when computing averages
    Examples::
        >>> ema = EMA(module.parameters(), decay=0.995)
        >>> # Train for a few epochs
        >>> for _ in range(epochs):
        >>>     # 训练过程中，更新完参数后，同步update shadow weights
        >>>     optimizer.step()
        >>>     ema.update(module.parameters())
        >>> # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
        >>> ema.store(module.parameters())
        >>> ema.copy_to(module.parameters())
        >>> # evaluate
        >>> ema.restore(module.parameters())
    Reference:
        [1]  https://github.com/fadel/pytorch_ema
    """  # noqa: ignore flake8"

    def __init__(
            self,
            parameters,
            decay,
            use_num_updates=True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)


def swa(args, model, model_path_list=None, start=2, end=6):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    if model_path_list is None:
        model_path_list = os.listdir(args.model_storage_dir)
        model_path_list = [args.model_storage_dir + '/' + model_name for model_name in model_path_list]
        tmp = []
        for model_path in model_path_list:
            if '.ipynb_checkpoints' not in model_path and f'epoch' in model_path:
                tmp.append(model_path)
        model_path_list = tmp
    else:
        start = 0
        end = len(model_path_list)
    print(model_path_list)

    assert 0 <= start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[start:end]:
            print(_ckpt)
            # logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1
    swa_model.to(args.device)
    return swa_model


class FGM(object):
    '''
    Example
    # 初始化
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in raw_data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''

    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                #                 print(name)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AWP:
    def __init__(
            self,
            model,
            optimizer,
            adv_param="weight",
            adv_lr=1,
            adv_eps=0.0001,
            start_epoch=0,
            adv_step=1,
            scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, input_ids, label_ids, epoch=0, scaler=None):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            if scaler is not None:
                with autocast():
                    loss_adv = self.model(input_ids=input_ids, labels=label_ids)[0]
                scaler.scale(loss_adv).backward()
            else:
                loss_adv = self.model(input_ids=input_ids, labels=label_ids)[0]
                loss_adv.backward()
            self.optimizer.zero_grad()
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.raw_data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
