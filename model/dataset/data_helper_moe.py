import os
import torch
import numpy as np
from PIL import Image
from functools import partial
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


def create_test_dataloader(args, processor, tokenizer, test_path, transformer, types=None):
    data_path = []
    types_list = []
    file_names = []
    # 在这里用for循环进行每张图片的预测只是一个示例，选手可以根据模型一次读入多张图片并进行预测
    for text_filename in os.listdir(test_path):
        text_img_path = os.path.join(test_path, text_filename)  # 每一张文字图片路径
        data_path.append(text_img_path)
        if types == 'text':
            types_list.append(1)
        elif types == 'num':
            types_list.append(0)
        file_names.append(text_filename)
    # print(data_path)
    val_dataset = OCRDataset(paths=data_path, processor=processor, tokenizer=tokenizer,
                             max_target_length=args.max_target_length,
                             transformer=transformer, mode='test', types=types_list, num_priori=args.num_priori,
                             text_priori=args.text_priori)

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
    return val_dataloader, file_names


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

    def __init__(self, paths, processor, tokenizer, max_target_length=128, transformer=lambda x: x, mode='train',
                 types=None, num_priori=None, text_priori=None):
        self.paths = paths
        self.processor = processor
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_target_length = max_target_length
        self.mode = mode
        self.types = types
        self.num_priori = num_priori
        self.text_priori = text_priori

    def __len__(self):
        return len(self.paths)

    def process_image_2_pixel_value(self, x: str):
        image = Image.open(x)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        if self.types[0] == 0:
            # num图像处理
            h, w = image.height, image.width
            img = np.array(image)
            angle = Image.ROTATE_90
            if h > w:
                top_black = (img[0:120, :, :] < 100).sum()
                down_black = (img[-120:, :, :] < 100).sum()
                if top_black < down_black:
                    angle = Image.ROTATE_270
                image = image.transpose(angle)
        image = self.transformer(image)  ##图像增强函数
        res = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return res

    def process_text_2_input_id(self, x: str):
        res = self.tokenizer(text=x, max_length=32, truncation=True, padding="max_length")['input_ids']
        return res

    def __getitem__(self, idx):
        image_file = self.paths[idx]

        if self.mode == 'test':
            text = ''
        else:
            # text = self.paths[idx][1]
            text = ' '.join(self.paths[idx][1:])

        pixel_value = self.process_image_2_pixel_value(image_file)
        if self.types is None:
            if self.mode == 'test':
                priori = None
            else:
                priori = [0, 3, 0.3, 0.4]
        else:
            if self.types[idx] == 0:  # num
                priori = self.num_priori
            else:  # text
                priori = self.text_priori
        return pixel_value.squeeze(), text, priori

    def pad_collate(self, batch):
        data = {}
        pixel_values, texts, priori = zip(*batch)
        tokenizer_output = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_target_length, padding=True, truncation=True, return_tensors='pt'
        )
        labels, _ = tokenizer_output.input_ids, tokenizer_output.attention_mask

        # for i in range(len(labels)):
        #     label = [ids if ids != self.processor.tokenizer.pad_token_id else -100 for ids in labels[i]]
        #     labels[i] = torch.LongTensor(label)

        data['pixel_values'] = torch.stack(pixel_values)
        data['labels'] = torch.LongTensor(labels)
        data['texts'] = texts
        data['priori'] = torch.Tensor(priori)

        return data
