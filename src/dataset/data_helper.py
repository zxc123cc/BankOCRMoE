import argparse
import csv
from typing import Any, Optional, Tuple
import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BertTokenizer
import random
from PIL import Image
from functools import partial
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torchvision.transforms as transforms

def create_dataloaders(args,processor,tokenizer,mode='train'):
    train_datas, dev_datas = [], []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))

    with open(args.dev_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            dev_datas.append(line.strip().split(" "))

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

    # train_transforms = transforms.Compose([
    #     transforms.ColorJitter(brightness=.5, hue=.3),
    #     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # ])

    train_dataset = OCRDataset(paths=train_datas, processor=processor, tokenizer=tokenizer,max_target_length=args.max_target_length,
                               transformer=train_transformer,mode=mode)
    val_dataset = OCRDataset(paths=dev_datas, processor=processor,tokenizer=tokenizer, max_target_length=args.max_target_length,
                              transformer=transformer,mode=mode)
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler,
                                        drop_last=False,
                                        collate_fn=train_dataset.pad_collate)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False,
                                      collate_fn=val_dataset.pad_collate)
    return train_dataloader, val_dataloader


def create_test_dataloader(args,processor,tokenizer,test_path,test_tmp_path):
    with open(test_tmp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    test_datas = []
    for line in lines[1:]:
        file_name = line.strip().split(",")[0]
        path = os.path.join(test_path, file_name)
        test_datas.append([path,''])

    transformer = lambda x: x  ##图像数据增强函数，可自定义

    val_dataset = OCRDataset(paths=test_datas, processor=processor,tokenizer=tokenizer, max_target_length=args.max_target_length,
                             transformer=transformer,mode='test')
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    val_sampler = SequentialSampler(val_dataset)

    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False,
                                      collate_fn=val_dataset.pad_collate)
    return val_dataloader


class OCRDataset(Dataset):
    """
    trocr 训练数据集处理
    文件数据结构
    /tmp/0/0.jpg #image
    /tmp/0/0.txt #text label
    ....
    /tmp/100/10000.jpg #image
    /tmp/100/10000.txt #text label
    """
    def __init__(self, paths, processor, tokenizer, max_target_length=128, transformer=lambda x:x,mode='train'):
        self.paths = paths
        self.processor = processor
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_target_length = max_target_length
        self.mode = mode

    def __len__(self):
        return len(self.paths)
    def process_image_2_pixel_value(self, x:str):
        image = Image.open(x)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = self.transformer(image) ##图像增强函数
        res = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return res

    def process_text_2_input_id(self, x:str) :
        res = self.tokenizer(text=x,max_length=32, truncation=True,padding="max_length")['input_ids']
        return res

    def __getitem__(self, idx):
        image_file = self.paths[idx][0]

        if self.mode == 'test':
            text = ''
        else:
            # text = self.paths[idx][1]
            text = ' '.join(self.paths[idx][1:])

        pixel_value = self.process_image_2_pixel_value(image_file)
        # labels = self.process_text_2_input_id(text)
        # encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        # return encoding
        return pixel_value.squeeze(), text

    def pad_collate(self, batch):
        data = {}
        pixel_values, texts = zip(*batch)
        tokenizer_output = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_target_length, padding=True, truncation=True,return_tensors='pt'
        )
        labels, _ = tokenizer_output.input_ids, tokenizer_output.attention_mask
        data['pixel_values'] = torch.stack(pixel_values)
        data['labels'] = torch.LongTensor(labels)
        data['texts'] = texts

        return data