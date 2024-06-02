import os
import argparse
from glob import glob
from dataset.dataset import trocrDataset, decode_text
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoConfig
from datasets import load_metric
import torchvision.transforms as transforms


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
    # print(label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    print([pred_str[0], label_str[0]])
    acc = sum(acc) / (len(acc) + 0.000001)

    with open('../pretrain_models/trocr-base-hand-cn-v2/results.txt', 'a+', encoding='utf-8') as f:
        f.write(f"cer:{cer}, acc:{acc}\n")

    return {"cer": cer, "acc": acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--cust_data_init_weights_path', default='../pretrain_models/trocr-base-stage1-cn-v2', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--checkpoint_path', default='../pretrain_models/trocr-base-hand-cn-v2', type=str,
                        help="训练模型保存地址")
    parser.add_argument('--dataset_path', default='../dataset/cust-data/*/*.jpg', type=str, help="训练数据集")
    parser.add_argument('--per_device_train_batch_size', default=32, type=int, help="train batch size")
    parser.add_argument('--per_device_eval_batch_size', default=8, type=int, help="eval batch size")
    parser.add_argument('--max_target_length', default=80, type=int, help="训练文字字符数")

    parser.add_argument('--num_train_epochs', default=50, type=int, help="训练epoch数")
    parser.add_argument('--eval_steps', default=1000, type=int, help="模型评估间隔数")
    parser.add_argument('--save_steps', default=1000, type=int, help="模型保存间隔步数")

    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0', type=str, help="GPU设置")

    args = parser.parse_args()
    print("train param")
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    print("loading data .................")
    paths = glob(args.dataset_path)

    # train_paths, test_paths = train_test_split(paths, test_size=0.05, random_state=10086)
    # print("train num:", len(train_paths), "test num:", len(test_paths))
    train_datas, test_datas = [], []
    with open('../datas/cust_data/pretrain_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip().split("\t")) == 2:
                train_datas.append(line.strip().split("\t"))

    with open('../datas/cust_data/pretrain_dev.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip().split("\t")) == 2:
                if len(line.strip().split("\t")[1]) != 0:
                    test_datas.append(line.strip().split("\t"))

    # print("train num:", len(train_paths), "test num:", len(test_paths))
    print("train num:", len(train_datas), "test num:", len(test_datas))

    ##图像预处理
    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}

    transformer = transforms.RandomChoice([
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

    cust_config = AutoConfig.from_pretrained(args.cust_data_init_weights_path)
    model_tmp = VisionEncoderDecoderModel.from_pretrained('../pretrain_models/trocr-base-stage1')
    # encoder = model_tmp.encoder
    model = VisionEncoderDecoderModel(cust_config, encoder=model_tmp.encoder)

    # model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 256
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 2

    cer_metric = load_metric("./cer.py")

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        fp16=True,
        output_dir=args.checkpoint_path,
        logging_steps=1000,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_total_limit=40,
        dataloader_num_workers=10,
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
