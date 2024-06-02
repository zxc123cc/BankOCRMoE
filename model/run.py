# run.py
import os, sys

sub_root = os.path.abspath(__file__).replace("run.py", "")
os.chdir(f'{sub_root}')
# 压缩预测文件  此函数不要修改！！！
import torch

# from inference_ensemble import predict_text
from inference_ensemble_num import predict_num
from inference_two_stage import predict_text


class Config():
    pretrain_model_dir = "./pretrain"
    max_target_length = 128
    num_priori = [0.2, 0.4, 0.4]
    text_priori = [0.4, 0.2, 0.4]
    num_workers = 6
    prefetch = 12
    val_batch_size = 8
    infer_num_beams = 8
    infer_max_length = 80
    infer_no_repeat_ngram_size = 5
    infer_min_length = 3
    length_penalty = 2.4
    repetition_penalty = 0.8


def parse_and_zip(result_save_path):
    os.system('zip -r submit.zip submit/')
    os.system(f'cp submit.zip {result_save_path}')


# 主函数，这个函数整体可根据选手自己模型的情况进行更改，在这里只是给出简单的示例
def main(to_pred_dir, result_save_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    args = Config()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.device)

    # 以下内容为示例，可以按照自己需求更改，主要是读入数据，并预测数据
    dirpath = os.path.abspath(to_pred_dir)  # 待预测文件夹路径
    text_path = os.path.join(dirpath, 'text')  # 待预测文字图片文件夹路径
    # print(args)
    # 预测text
    text_result_df = predict_text(args, text_path)

    num_path = os.path.join(dirpath, 'num')  # 待预测数字图片文件夹路径
    num_result_df = predict_num(args, num_path)

    # ！！！注意这里一定要新建一个submit文件夹，然后把你预测的结果存入这个文件夹中，这两个预测结果文件的命名不可修改，需要严格和下面写出的一致。
    # os.mkdir('submit')
    os.makedirs('submit', exist_ok=True)
    text_result_df[['file_name', 'result']].to_csv('submit/text_sub.csv', encoding='utf8', index=False, )
    num_result_df[['file_name', 'result']].to_csv('submit/num_sub.csv', encoding='utf8', index=False, )

    parse_and_zip(result_save_path)


if __name__ == "__main__":
    # ！！！以下内容不允许修改，修改会导致评分出错
    to_pred_dir = sys.argv[1]  # 待预测数据存放路径
    result_save_path = sys.argv[2]  # 预测结果存放路径
    main(to_pred_dir, result_save_path)
