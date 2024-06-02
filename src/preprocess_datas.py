import os
from PIL import Image
import numpy as np

# os.makedirs(save_path, exist_ok=True)
#
#
# for file_name in os.listdir(data_path):
#     file_path = os.path.join(data_path, file_name)
#     if file_name.endswith(".jpg") or file_name.endswith(".png"):
#         image = Image.open(os.path.join(data_path, file_name)).convert("RGB")
#         h, w = image.height, image.width
#         img = np.array(image)
#         angle = Image.ROTATE_90
#         if h > w:
#             top_black = (img[0:120, :, :] < 100).sum()
#             down_black = (img[-120:, :, :] < 100).sum()
#             if top_black < down_black:
#                 angle = Image.ROTATE_270
#
#             image = image.transpose(angle)
#         image.save(os.path.join(save_path, file_name))

def preprocess(save_path,data_path):
    os.makedirs(save_path, exist_ok=True)
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image = Image.open(os.path.join(data_path, file_name)).convert("RGB")
            h, w = image.height, image.width
            img = np.array(image)
            angle = Image.ROTATE_90
            if h > w:
                top_black = (img[0:120, :, :] < 100).sum()
                down_black = (img[-120:, :, :] < 100).sum()
                if top_black < down_black:
                    angle = Image.ROTATE_270

                image = image.transpose(angle)
            image.save(os.path.join(save_path, file_name))


if __name__ == '__main__':
    preprocess(data_path = '../datas/forUser/train/num',
               save_path = '../datas/tmp_data/train/pre_num')
    preprocess(data_path = '../datas/forUser/testA/num',
               save_path = '../datas/tmp_data/testA/pre_num')
