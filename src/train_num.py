import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
from glob import glob
from dataset.data_helper2 import trocrDataset, decode_text
from transformers import TrOCRProcessor
# from transformers import VisionEncoderDecoderModel
from models.modeling_vision_encoder_decoder import VisionEncoderDecoderModel
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

    return {"cer": 1-cer, "acc": acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--cust_data_init_weights_path', default='../pretrain_models/pretrain_910000', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--checkpoint_path', default='../model_storage/num_seed2023_adda_all', type=str, help="训练模型保存地址")
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
    parser.add_argument('--seed', type=int, default=2023)

    args = parser.parse_args()
    print("train param")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    set_random_seed(args.seed)
    print("loading data .................")
    paths = glob(args.dataset_path)

    train_datas, test_datas = [], []

    # train.............

    with open('../datas/cust_data/train_num1.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:]:
            train_datas.append(line.strip().split(" "))

    with open('../datas/cust_data/testA_num_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:]:
            train_datas.append(line.strip().split(" "))

    with open('../datas/cust_data/dev_num1.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))


    # dev.............
    with open('../datas/cust_data/dev_num1.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            test_datas.append(line.strip().split(" "))


    # print("train num:", len(train_paths), "test num:", len(test_paths))
    print("train num:", len(train_datas), "test num:", len(test_datas))

    ##图像预处理
    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()
    print("vocab:", len(vocab))
    vocab_inp = {vocab[key]: key for key in vocab}

    transformer = lambda x: x  ##图像数据增强函数，可自定义
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
                                 transformer=transformer)
    transformer = lambda x: x  ##图像数据增强函数
    eval_dataset = trocrDataset(paths=test_datas, processor=processor, max_target_length=args.max_target_length,
                                transformer=transformer)

    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path,mode=args.loss_mode)

    device = torch.device('cuda')
    model.to(device)
    if args.loss_mode == 'emo':
        model.cost_embedding = model.cost_embedding.to(device)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.vocab_size = model.config.decoder.vocab_size

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
        save_total_limit=5,
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

