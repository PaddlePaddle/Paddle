# TSM 视频分类模型

---

## 内容

- [模型简介](#模型简介)
- [快速开始](#快速开始)
- [参考论文](#参考论文)


## 模型简介

Temporal Shift Module是由MIT和IBM Watson AI Lab的Ji Lin，Chuang Gan和Song Han等人提出的通过时间位移来提高网络视频理解能力的模块，其位移操作原理如下图所示。

<p align="center">
<img src="./images/temporal_shift.png" height=250 width=800 hspace='10'/> <br />
Temporal shift module
</p>

上图中矩阵表示特征图中的temporal和channel维度，通过将一部分的channel在temporal维度上向前位移一步，一部分的channel在temporal维度上向后位移一步，位移后的空缺补零。通过这种方式在特征图中引入temporal维度上的上下文交互，提高在时间维度上的视频理解能力。

TSM模型是将Temporal Shift Module插入到ResNet网络中构建的视频分类模型，本模型库实现版本为以ResNet-50作为主干网络的TSM模型。

详细内容请参考论文[Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1)

## 快速开始

### 安装说明

#### paddle安装

   本项目依赖于 PaddlePaddle 1.7及以上版本或适当的develop版本，请参考 [安装指南](http://www.paddlepaddle.org/#quick-start) 进行安装

#### 代码下载及环境变量设置

    克隆代码库到本地，并设置`PYTHONPATH`环境变量
    ```shell
    git clone https://github.com/PaddlePaddle/hapi
    cd hapi
    export PYTHONPATH=$PYTHONPATH:`pwd`
    cd tsm
    ```

### 数据准备

TSM的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。数据下载及准备请参考[数据说明](./dataset/README.md)

#### 小数据集验证

为了便于快速迭代，我们采用了较小的数据集进行动态图训练验证，从Kinetics-400数据集中选取分类标签(label)分别为 0, 2, 3, 4, 6, 7, 9, 12, 14, 15的即前10类数据验证模型精度。

### 模型训练

数据准备完毕后，可使用`main.py`脚本启动训练和评估，如下脚本会自动每epoch交替进行训练和模型评估，并将checkpoint默认保存在`tsm_checkpoint`目录下。

`main.py`脚本参数可通过如下命令查询

```shell
python main.py --help
```

#### 静态图训练

使用如下方式进行单卡训练:

```shell
export CUDA_VISIBLE_DEVICES=0
python main.py --data=<path/to/dataset> --batch_size=16
```

使用如下方式进行多卡训练:

```shell
CUDA_VISIBLE_DEVICES=0,1 python main.py --data=<path/to/dataset> --batch_size=8
```

#### 动态图训练

动态图训练只需要在运行脚本时添加`-d`参数即可。

使用如下方式进行单卡训练:

```shell
export CUDA_VISIBLE_DEVICES=0
python main.py --data=<path/to/dataset> --batch_size=16 -d
```

使用如下方式进行多卡训练:

```shell
CUDA_VISIBLE_DEVICES=0,1 python main.py --data=<path/to/dataset> --batch_size=8 -d
```

**注意：** 对于静态图和动态图，多卡训练中`--batch_size`为每卡上的batch_size，即总batch_size为`--batch_size`乘以卡数

### 模型评估

可通过如下两种方式进行模型评估。

1. 自动下载Paddle发布的[TSM-ResNet50](https://paddlemodels.bj.bcebos.com/hapi/tsm_resnet50.pdparams)权重评估

```
python main.py --data<path/to/dataset> --eval_only
```

2. 加载checkpoint进行精度评估

```
python main.py --data<path/to/dataset> --eval_only --weights=tsm_checkpoint/final
```

#### 评估精度

在10类小数据集下训练模型权重见[TSM-ResNet50](https://paddlemodels.bj.bcebos.com/hapi/tsm_resnet50.pdparams)，评估精度如下：

|Top-1|Top-5|
|:-:|:-:|
|76.5%|98.0%|

## 参考论文

- [Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383v1), Ji Lin, Chuang Gan, Song Han

