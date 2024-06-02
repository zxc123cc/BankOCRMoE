import os
import random

random.seed(2023)


def load_labels(path: str, prefix_path):
    datas = []
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split(",")
            datas.append([prefix_path + line[0], line[1]])
    return datas


def split_num_data():
    # 1、构造vocab_text.txt

    train_data = []
    with open('../datas/forUser/train/train_num.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip().split(",")
        if len(line) == 2:
            train_data.append([os.path.join("../datas/tmp_data/train/pre_num/", line[0]), line[1]])

    num = 4000
    with open('../datas/cust_data/train_num1.txt', 'w', encoding='utf-8') as f:
        for data in train_data[0:num]:
            data = " ".join(data)
            f.write(data + "\n")

    with open('../datas/cust_data/dev_num1.txt', 'w', encoding='utf-8') as f:
        for data in train_data[num:]:
            data = " ".join(data)
            f.write(data + "\n")


def split_text_data_():
    # 1、构造vocab_text.txt

    train_data = []
    with open('../datas/forUser/train/train_text.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip().split(",")
        train_data.append([os.path.join("../datas/forUser/train/text/", line[0]), ",".join(line[1:])])
    print(len(train_data))
    with open('../datas/cust_data/train_text.txt', 'w', encoding='utf-8') as f:
        for data in train_data:
            data = " ".join(data)
            f.write(data + "\n")


def split_testa_text():
    train_data,val_data = [],[]
    val_num,num_x,num_y = 200,0,0
    with open('../datas/forUser/testA/ans_text_a.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip().split(",")
        if ' ' in ",".join(line[1:]):
            num_x += 1
        else:
            num_y += 1

    ratio = num_x / num_y

    num_a = int((val_num*ratio) / (1+ratio))
    num_b = val_num - num_a
    print(num_a,num_b)
    for line in lines[1:]:
        line = line.strip().split(",")
        if ' ' in line[1] and num_a > 0:
            num_a -= 1
            val_data.append([os.path.join("../datas/forUser/testA/text/", line[0]), ",".join(line[1:]) ])
        elif num_b > 0:
            num_b -= 1
            val_data.append([os.path.join("../datas/forUser/testA/text/", line[0]), ",".join(line[1:]) ])
        else:
            train_data.append([os.path.join("../datas/forUser/testA/text/", line[0]), ",".join(line[1:]) ])

    with open('../datas/cust_data/testA_text_train.txt', 'w', encoding='utf-8') as f:
        for data in train_data:
            data = " ".join(data)
            f.write(data + "\n")

    with open('../datas/cust_data/testA_text_dev.txt', 'w', encoding='utf-8') as f:
        for data in val_data:
            data = " ".join(data)
            f.write(data + "\n")
    print(len(train_data),len(val_data))


def split_testa_num():
    # 1、构造vocab_text.txt

    train_data = []
    with open('../datas/forUser/testA/ans_num_a.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip().split(",")
        if len(line) == 2:
            train_data.append([os.path.join("../datas/tmp_data/testA/pre_num/", line[0]), line[1]])

    with open('../datas/cust_data/testA_num_train.txt', 'w', encoding='utf-8') as f:
        for data in train_data:
            data = " ".join(data)
            f.write(data + "\n")

if __name__ == '__main__':
    split_num_data()
    split_text_data_()

    split_testa_text()
    split_testa_num()
