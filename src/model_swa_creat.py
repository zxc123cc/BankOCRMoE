import copy
from collections import defaultdict
import numpy as np
import math
import sys
sys.path.append('models')
import six
import torch
import os
from transformers import AutoTokenizer,TrOCRProcessor
from models.modeling_vision_encoder_decoder import VisionEncoderDecoderModel
from dataset.data_helper import create_dataloaders
from models.model_moe import BankOCRMoE
from transformers import VisionEncoderDecoderConfig
import argparse

def get_model_path_list(base_dir,swa_start,swa_end):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []
    for i in range(swa_start,swa_end+1):
        model_lists.append(base_dir + '/model_epoch{}.bin'.format(i))
    # for root, dirs, files in os.walk(base_dir):
    #     for _file in files:
    #         if '.bin' in _file and 'swa' not in _file and 'model_best.bin' not in _file:
    #             model_lists.append(os.path.join(root, _file).replace("\\", '/'))
    # model_lists = sorted(model_lists, key=lambda x: int(x.split('/')[-1].split('_')[1][5:]))
    return model_lists


def get_swa(args,model_type,model_path_list,ckpt_file):
    # 1. load data
    device = torch.device(args.device)
    print(device)
    processor = TrOCRProcessor.from_pretrained(args.pretrain_model_dir)
    # 2. build model and optimizers
    if model_type != 'moe':
        model = VisionEncoderDecoderModel.from_pretrained(args.pretrain_model_dir)
    else:
        config = VisionEncoderDecoderConfig.from_pretrained(args.pretrain_model_dir)
        model = BankOCRMoE(config=config)
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    print(model_path_list)
    num = len(model_path_list)
    state_list = []
    for temp in model_path_list:
        state = torch.load(temp)
        state_list.append(state)

    for pkey in state_list[0]:
        temp = 0.0
        for k in range(len(state_list)):
            temp = temp + state_list[k][pkey] / num
        state_list[0][pkey] = temp

    msg = model.load_state_dict(state_list[0], strict=False)
    print(msg)

    # 5. save checkpoint
    model.half()
    torch.save(model.state_dict(),ckpt_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--pretrain_model_dir', default='../pretrain_models/pretrain_910000', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help="GPU设置")
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model_type', default='moe', type=str)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #moe
    model_path_list = [
        '../model_storage/pretrain_910000_moe_seed2023_adda_addmerge_all/checkpoint-4500/pytorch_model.bin',
        '../model_storage/pretrain_910000_moe_seed2023_adda_addmerge_all/checkpoint-5000/pytorch_model.bin',
        '../model_storage/pretrain_910000_moe_seed2023_adda_addmerge_all/checkpoint-5500/pytorch_model.bin',
        '../model_storage/pretrain_910000_moe_seed2023_adda_addmerge_all/checkpoint-6000/pytorch_model.bin',
    ]
    get_swa(args,model_type='moe',model_path_list=model_path_list,
            ckpt_file='../model_storage/moe_seed2023_adda_addmerge_all.bin')

    #text seed 2023
    model_path_list = [
        '../model_storage/text_seed2023_adda_addmergeA_all/checkpoint-1750/pytorch_model.bin',
        '../model_storage/text_seed2023_adda_addmergeA_all/checkpoint-2000/pytorch_model.bin',
        '../model_storage/text_seed2023_adda_addmergeA_all/checkpoint-2250/pytorch_model.bin',
        '../model_storage/text_seed2023_adda_addmergeA_all/checkpoint-2500/pytorch_model.bin',
    ]
    get_swa(args,model_type='En-De',model_path_list=model_path_list,
            ckpt_file='../model_storage/text_seed2023_adda_addmergeA_all.bin')

    #text seed 42
    model_path_list = [
        '../model_storage/text_seed42_adda_addmergeA_all/checkpoint-1750/pytorch_model.bin',
        '../model_storage/text_seed42_adda_addmergeA_all/checkpoint-2000/pytorch_model.bin',
        '../model_storage/text_seed42_adda_addmergeA_all/checkpoint-2250/pytorch_model.bin',
        '../model_storage/text_seed42_adda_addmergeA_all/checkpoint-2500/pytorch_model.bin',
    ]
    get_swa(args,model_type='En-De',model_path_list=model_path_list,
            ckpt_file='../model_storage/text_seed42_adda_addmergeA_all.bin')

    #text seed 3407
    model_path_list = [
        '../model_storage/text_seed3407_adda_addmergeA_all/checkpoint-1750/pytorch_model.bin',
        '../model_storage/text_seed3407_adda_addmergeA_all/checkpoint-2000/pytorch_model.bin',
        '../model_storage/text_seed3407_adda_addmergeA_all/checkpoint-2250/pytorch_model.bin',
        '../model_storage/text_seed3407_adda_addmergeA_all/checkpoint-2500/pytorch_model.bin',
    ]
    get_swa(args,model_type='En-De',model_path_list=model_path_list,
            ckpt_file='../model_storage/text_seed3407_adda_addmergeA_all.bin')

    #num seed 2023
    model_path_list = [
        '../model_storage/num_seed2023_adda_all/checkpoint-3000/pytorch_model.bin',
        '../model_storage/num_seed2023_adda_all/checkpoint-3500/pytorch_model.bin',
        '../model_storage/num_seed2023_adda_all/checkpoint-4000/pytorch_model.bin',
        '../model_storage/num_seed2023_adda_all/checkpoint-4500/pytorch_model.bin',
    ]
    get_swa(args,model_type='En-De',model_path_list=model_path_list,
            ckpt_file='../model_storage/num_seed2023_adda_all.bin')