import os
from tqdm import tqdm
import random
random.seed(2023)

data_path = '../datas/chinese_ocr'

datas = []
error_count = 0
count = 0
vocabs = set()
for r, _, files in tqdm(os.walk(data_path)):
    count += 1
    # if count > 1000:
    #     break
    for fname in tqdm(files):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            name = fname.split('.')[0]
            try:
                with open(os.path.join(r, name + ".txt"), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                text = ""
                for line in lines:
                    line = line.strip()
                    for l in line:
                        vocabs.add(l)
                    text = text + line.replace(" ", "") + "\n"
                text = text.strip()

                datas.append([os.path.join(r, fname), text])
            except:
                error_count += 1


random.shuffle(datas)

dev_num = 10000

with open('../datas/cust_data/pretrain_train.txt', 'w', encoding='utf-8') as f:
    for data in datas[0:-1 * dev_num]:
        data = "\t".join(data)
        f.write(data + "\n")

print("训练集:", len(datas[0:-1 * dev_num]))


with open('../datas/cust_data/pretrain_dev.txt', 'w', encoding='utf-8') as f:
    for data in datas[-1 * dev_num:]:
        data = "\t".join(data)
        f.write(data + "\n")


with open('../datas/cust_data/pretrain_vocab.txt', 'w', encoding='utf-8') as f:
    for vocab in vocabs:
        f.write(vocab + "\n")

print("验证集:", len(datas[-1 * dev_num:]))
print("总数量:", len(datas))
print("error_count:", error_count)
print("vocabs:", len(vocabs))

