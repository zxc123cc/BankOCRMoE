import os
import random

from PIL import Image
from util.utils import set_random_seed
#两张图片水平拼接
def merge_images(image1_path, image2_path, output_path,merge_type='col'):
    # 打开两张图片
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 获取第一张图片的尺寸
    width1, height1 = image1.size
    width2, height2 = image2.size

    bili = 0.9
    if merge_type == 'row':
        if (height2<height1  and height2 / height1 < bili) or (height1<height2  and height1 / height2 < bili):
            return 0

    if merge_type == 'col':
        if (width2<width1  and width2 / width1 < bili) or (width1<width2  and width1 / width2 < bili):
            return 0
    # return 1
    # 调整第二张图片的尺寸与第一张相同
    image2 = image2.resize((width1, height1))
    if merge_type == 'row':
        # 创建一个新的空白图片，尺寸为两张图片横向拼接后的尺寸
        merged_image = Image.new('RGB', (width1 * 2, height1))
    if merge_type == 'col':
        # 创建一个新的空白图片，尺寸为两张图片纵向拼接后的尺寸
        merged_image = Image.new('RGB', (width1, height1*2))
    # merged_image.save("ttt.jpg")全黑图片

    # 将两张图片拼接到新图片上
    merged_image.paste(image1, (0, 0))#左上角
    if merge_type == 'row':
        merged_image.paste(image2, (width1, 0))
    if merge_type == 'col':
        merged_image.paste(image2, (0, height1))
    # 保存合成后的图片
    merged_image.save(output_path)
    return 1

# # 示例用法
# image1_path = '../datas/forUser/train/text/428.jpg'
# image2_path = '../datas/forUser/train/text/449.jpg'
# output_path = 'merged_image.jpg'
# merge_images(image1_path, image2_path, output_path)

def run_merge(output_image_dir,output_label_file,merge_type='col'):
    os.makedirs(output_image_dir, exist_ok=True)
    img_dir = '../datas/forUser/train/text/'

    with open('../datas/forUser/train/train_text.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    file_name_list = []
    label_list = []
    for line in lines[1:]:
        file_name = line.strip().split(",")[0]
        # path = os.path.join(test_path, file_name)
        label = line.strip().split(",")[1]
        file_name_list.append(file_name)
        label_list.append(label)

    fw = open(output_label_file, 'w', encoding='utf-8')
    for i in range(len(file_name_list)):
        for j in range(len(file_name_list)):
            file_name1 = file_name_list[i]
            file_name2 = file_name_list[j]
            label1 = label_list[i]
            label2 = label_list[j]
            if ' ' in label1 or ' ' in label2:
                print(2)
                continue

            image1_path = img_dir + file_name1
            image2_path = img_dir + file_name2
            output_path = output_image_dir + file_name1[:-4] + '_' + file_name2[:-4] + '.jpg'
            flag = merge_images(image1_path, image2_path, output_path,merge_type=merge_type)
            if flag:
                print(1)
                fw.write(' '.join([output_path,label1,label2]) + '\n')


def run_merge2(output_image_dir,output_label_file,merge_type='col'):
    os.makedirs(output_image_dir, exist_ok=True)
    train_datas = []
    with open('../datas/cust_data/train_text.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
    with open('../datas/cust_data/testA_text_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
    with open('../datas/cust_data/testA_text_dev.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
    # random.shuffle(train_datas)
    image_name_list = []
    label_list = []
    path_list = []
    for data in train_datas:
        path = data[0]
        path_list.append(path)
        image_name = path.split('/')[-1][:-4]
        image_name_list.append(image_name)
        label = ' '.join(data[1:])
        label_list.append(label)

    num = 0
    label_dict = {}
    fw = open(output_label_file, 'w', encoding='utf-8')
    for i in range(len(path_list)):
        for j in range(len(path_list)):
            label1 = label_list[i]
            label2 = label_list[j]
            if label1 == label2:
                continue
            if ' ' in label1 or ' ' in label2:
                continue
            if len(label1) <= 5 or len(label2) <= 5 or len(label1) > 18 or len(label2) > 18:
                continue
            if len(label2) > len(label1):
                continue
            if (label1 in label_dict and label_dict[label1] >= 5) or (label2 in label_dict and label_dict[label2] >= 5):
                continue
            output_path = output_image_dir + image_name_list[i] + '_' + image_name_list[j] + '.jpg'
            flag = merge_images(path_list[i], path_list[j], output_path,merge_type=merge_type)
            if flag:
                fw.write(' '.join([output_path,label1,label2]) + '\n')
                num += 1
                if label1 not in label_dict:
                    label_dict[label1] = 0
                if label2 not in label_dict:
                    label_dict[label2] = 0
                label_dict[label1] += 1
                label_dict[label2] += 1
                print(num)


def run_merge3(output_image_dir,output_label_file,merge_type='col'):
    os.makedirs(output_image_dir, exist_ok=True)
    train_datas = []
    with open('../datas/cust_data/train_text.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))
    with open('../datas/cust_data/testA_text_train.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            train_datas.append(line.strip().split(" "))

    image_name_list = []
    label_list = []
    path_list = []
    for data in train_datas:
        path = data[0]
        path_list.append(path)
        image_name = path.split('/')[-1][:-4]
        image_name_list.append(image_name)
        label = ' '.join(data[1:])
        label_list.append(label)

    num = 0
    label_dict = {}
    fw = open(output_label_file, 'w', encoding='utf-8')
    for i in range(len(path_list)):
        for j in range(len(path_list)):
            label1 = label_list[i]
            label2 = label_list[j]
            if label1 == label2:
                continue
            if ' ' in label1 or ' ' in label2:
                continue
            if len(label1) <= 3 or len(label2) <= 3 or len(label1) > 12 or len(label2) > 12:
                continue
            if (label1 in label_dict and label_dict[label1] >= 3) or (label2 in label_dict and label_dict[label2] >= 3):
                continue
            output_path = output_image_dir + image_name_list[i] + '_' + image_name_list[j] + '.jpg'
            flag = merge_images(path_list[i], path_list[j], output_path,merge_type=merge_type)
            if flag:
                fw.write(' '.join([output_path,label1,label2]) + '\n')
                num += 1
                if label1 not in label_dict:
                    label_dict[label1] = 0
                if label2 not in label_dict:
                    label_dict[label2] = 0
                label_dict[label1] += 1
                label_dict[label2] += 1
                print(num)



if __name__ == '__main__':
    # run_merge(output_image_dir='../datas/tmp_data/merge_text/',output_label_file='../datas/cust_data/train_text_merge.txt')
    # run_merge(output_image_dir='../datas/tmp_data/merge_text_row/',output_label_file='../datas/cust_data/train_text_merge_row.txt',
    #           merge_type='row')
    run_merge2(output_image_dir='../datas/tmp_data/merge_text_addA_col/',output_label_file='../datas/cust_data/train_text_merge_addA_col_all.txt',
               merge_type='col')
    # run_merge2(output_image_dir='../datas/tmp_data/merge_text_addA_col/',output_label_file='../datas/cust_data/train_text_merge_addA_col_all_3407.txt',
    #            merge_type='col')
    # run_merge3(output_image_dir='../datas/tmp_data/merge_text_addA_row/',output_label_file='../datas/cust_data/train_text_merge_addA_row.txt',
    #            merge_type='row')