import os
import sys
sys.path.append('models')
os.environ["WANDB_DISABLED"] = "true"
import argparse
from glob import glob
from dataset.data_helper_moe2 import trocrDataset, decode_text
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderConfig
# from transformers import VisionEncoderDecoderModel
from models.model_moe import BankOCRMoE
from transformers import default_data_collator
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
# from trainer.trainer_seq2seq import Seq2SeqTrainer
# from trainer.training_args_seq2seq import Seq2SeqTrainingArguments
from datasets import load_metric
import torchvision.transforms as transforms
import torch
from util.utils import set_random_seed

def compute_metrics(pred):
    """
    计算cer,acc
    :param pred:
    :return:
    """
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = [decode_text(pred_id, vocab, vocab_inp) for pred_id in pred_ids]
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = [decode_text(labels_id, vocab, vocab_inp) for labels_id in labels_ids]
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    print([pred_str[0], label_str[0]])
    acc = sum(acc) / (len(acc) + 0.000001)

    return {"cer": cer, "acc": acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--pretrain_model_dir', default='../pretrain_models/pretrain_910000', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--checkpoint_path', default='../model_storage/pretrain_910000_moe_seed2023_adda_addmerge_all', type=str, help="训练模型保存地址")
    parser.add_argument('--dataset_path', default='./dataset/cust-data/*/*.jpg', type=str, help="训练数据集")
    parser.add_argument('--per_device_train_batch_size', default=8, type=int, help="train batch size")
    parser.add_argument('--per_device_eval_batch_size', default=8, type=int, help="eval batch size")
    parser.add_argument('--max_target_length', default=70, type=int, help="训练文字字符数")
    # 5, 1000,  2000  text
    # 5, 20, last  num
    # 5, 1000, 2000 all
    parser.add_argument('--num_train_epochs', default=5, type=int, help="训练epoch数")
    parser.add_argument('--eval_steps', default=500, type=int, help="模型评估间隔数")
    parser.add_argument('--save_steps', default=500, type=int, help="模型保存间隔步数")

    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help="GPU设置")

    parser.add_argument('--loss_mode', default='mle', type=str)
    parser.add_argument('--lmd', type=float, default=0.1)
    parser.add_argument('--num_priori', type=list, default=[0.2,0.4,0.4])
    parser.add_argument('--text_priori', type=list, default=[0.4,0.2,0.4])
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()
    print("train param")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    set_random_seed(args.seed)
    print("loading data .................")
    paths = glob(args.dataset_path)

    # train_paths, test_paths = train_test_split(paths, test_size=0.05, random_state=10086)
    # print("train num:", len(train_paths), "test num:", len(test_paths))
    train_datas, test_datas = [], []
    train_types, test_types = [], []

    # train.............
    with open('../datas/cust_data/train_num1.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:]:
            train_datas.append(line.strip().split(" "))
            train_types.append(0)
    with open('../datas/cust_data/testA_num_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:]:
            train_datas.append(line.strip().split(" "))
            train_types.append(0)
    with open('../datas/cust_data/dev_num1.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
            train_types.append(0)
            
    with open('../datas/cust_data/train_text_merge_addA_col_all.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:]:
            train_datas.append(line.strip().split(" "))
            train_types.append(1)
    with open('../datas/cust_data/train_text.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
            train_types.append(1)
    with open('../datas/cust_data/testA_text_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
            train_types.append(1)
    with open('../datas/cust_data/testA_text_dev.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
            train_types.append(1)

    # dev.............
    with open('../datas/cust_data/dev_num1.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            test_datas.append(line.strip().split(" "))
            test_types.append(0)
    with open('../datas/cust_data/testA_text_dev.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            test_datas.append(line.strip().split(" "))
            test_types.append(1)
            
    # print("train num:", len(train_paths), "test num:", len(test_paths))
    print("train num:", len(train_datas), "test num:", len(test_datas))

    ##图像预处理
    processor = TrOCRProcessor.from_pretrained(args.pretrain_model_dir)
    vocab = processor.tokenizer.get_vocab()
    print("vocab:", len(vocab))
    vocab_inp = {vocab[key]: key for key in vocab}

    transformer = lambda x: x  ##图像数据增强函数，可自定义
    # transformer = transforms.Grayscale(num_output_channels=3)
    train_transformer = transforms.RandomChoice([
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: x,
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.3),

        transforms.Grayscale(num_output_channels=3),

        transforms.RandomAffine(degrees=10),
        transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),

        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    ])

    train_dataset = trocrDataset(paths=train_datas, processor=processor, max_target_length=args.max_target_length,
                                 transformer=transformer,types=train_types,num_priori=args.num_priori,text_priori=args.text_priori)
    transformer = lambda x: x  ##图像数据增强函数
    eval_dataset = trocrDataset(paths=test_datas, processor=processor, max_target_length=args.max_target_length,
                                transformer=transformer,types=test_types,num_priori=args.num_priori,text_priori=args.text_priori)

    config = VisionEncoderDecoderConfig.from_pretrained(args.pretrain_model_dir)
    model = BankOCRMoE(config=config,num_experts=3)
    model.load_moe_encoder_decoder_from_vision_pretrain(args.pretrain_model_dir)

    device = torch.device('cuda')
    model.to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    
    model.config.max_length = 256
    model.config.min_length = 3
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 5
    model.config.length_penalty = 2.4
    model.config.num_beams = 8
    model.config.repetition_penalty = 0.8

    cer_metric = load_metric("./util/cer.py")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps", #or epoch
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        fp16=True,
        output_dir=args.checkpoint_path,
        logging_steps=100,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_total_limit=8,
        dataloader_num_workers=8,
        seed=args.seed
    )

    # seq2seq trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_path, 'last'))
    processor.save_pretrained(os.path.join(args.checkpoint_path, 'last'))

