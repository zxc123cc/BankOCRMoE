# 第四届厦门国际银行数创金融杯建模大赛冠军方案（含OCR预训练权重）
## 0.环境
cuda11.3

torch==1.9.1+cu111

transformers==4.33.3

也可直接运行命令：
```bash
pip install requirements.txt
```

## 1.预训练准备

### 1.1 ocr训练数据集
ocr ctc训练数据集(压缩包解码:chineseocr)  
百度网盘地址:链接: https://pan.baidu.com/s/1UcUKUUELLwdM29zfbztzdw 提取码: atwn   
gofile地址:http://gofile.me/4Nlqh/uT32hAjbx 密码 https://github.com/chineseocr/chineseocr

上述链接为从github开源项目获得，项目链接为：https://github.com/chineseocr/darknet-ocr

此外，本团队已经在gofile中上传了该部分数据，若上述链接无法获取数据，可直接在此处获取： https://gofile.io/d/Po6qfN

### 1.2 英文TrOCR-base权重
本团队使用了英文TrOCR-base权重中的Encoder部分对我们设计的中文TrOCR-base中的Encoder初始化

英文TrOCR-base权重地址：https://huggingface.co/microsoft/trocr-base-stage1

## 2.预训练
进入到src目录下，运行预训练脚本(以下皆是)
### 2.1 数据处理

```bash
sh scripts/pretrain_data_process.sh
```

### 2.2 大规模领域预训练

此处设置的多卡，可根据实际情况调整`scripts/pretrain.sh`里的CUDA_VISIBLE_DEVICES
```bash
sh scripts/pretrain.sh
```

注：由于数据量较大，该步骤耗时较久，我们团队提供了预训练好的权重供复现（存放在 `pretrain_models/pretrain_910000`）
链接：https://pan.baidu.com/s/17OtBQFG_KNR_6z76C9z_xw?pwd=47qe
提取码：47qe


## 2.数据处理
通过对任务一的数据集进行分析，我们发现该数据共四种形状，我们规定横向图像的形状为标准形状。在badcase,分析的时候发现第三类和第四类图像（竖着的）经过处理后变形较为严重，影响识别效果。
我们利用图像处理算法，首先识别到要识别的图像的形状，然后根据其与标准图像得角度进行旋转，将其变化为标准图像，提升了ocr的识别效果。

之后，我们将数据进行处理，弄成后续微调所需的数据形式,并划分训练验证集。

```bash
sh scripts/process_data.sh
```

## 3.微调阶段数据增强
由于数据中包含大量图像不同而 label 相同的样本，这与预训练阶段时使用的数据增强，即缩放、裁剪、翻转、旋转及变化颜色等操作等价，若在微调时继续采用这些操作则会引入大量噪声，进一步的挤压 hard sample 的空间。因此，我们应设计一种针对
此赛题 hard sample 的图像合成策略，来一定程度上抵消数据中 easy sample 带来的“过采样”问题。

```bash
sh scripts/merge_data.sh
```


## 4.微调
微调所有中间权重及最终权重全部存放于`model_storage`文件夹下
```bash
sh scripts/finetune.sh
```

使用swa平均多个局部最优点
```bash
sh scripts/swa.sh
```

## 5.推理
我们将B榜用于推理的所有文件（包括权重）全部放在`model`文件夹下，按照b榜情况推理即可

## 其他
更多详细内容请参考文档报告
