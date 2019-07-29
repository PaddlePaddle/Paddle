DyGraph模式下非对称量化
========

简介
--------
随着卷积神经网络模型堆叠的层数越来越多，网络模型的权重参数数量也随之增长，专用硬件平台可以很好的解决计算与存储的双重需求，但目前还不成熟，
存在些亟待解决的问题，没能大规模商用。对模型进行参数量化的主要目的是减小模型存储体积，加速运算，能够将算法应用在通用的嵌入式移动平台。

现在神经网络的weight和bias都是用单精度4字节的float32或者双精度8字节的float64的表示，为了降低模型的存储空间而达到模型压缩加速的目的，
越来越多的学者企图通过更少的bit位来表示模型实际的浮点型权值。

该方法采用非对称线性量化方法将weights和bias量化为uint8，权重的最大值和最小值根据每一层的权值范围确定，qmax=max(w), qmin=min(w)。


## 代码结构
```
.
├── models                          # 网络模型
│   └──resnet.py
├── quantization_toolbox            # 量化工具箱
│   └──asymmetric_quantization.py
├── resnet_params                   # 模型参数
└── quant_test.py                   # 量化Demo


```

## 使用的数据

教程中使用`paddle.dataset.flowers`数据集作为训练数据，该数据集通过`paddle.dataset`模块自动下载到本地。

## 训练测试Residual Network

在GPU单卡上训练Residual Network:

```
env CUDA_VISIBLE_DEVICES=0 python train.py
```

这里`CUDA_VISIBLE_DEVICES=0`表示是执行在0号设备卡上，请根据自身情况修改这个参数。

## 输出

```text
test | batch step 0, loss 0.000 acc1 0.469 acc5 0.781
test | batch step 10, loss 0.000 acc1 0.452 acc5 0.739
test | batch step 20, loss 0.000 acc1 0.439 acc5 0.741
test | batch step 30, loss 0.000 acc1 0.430 acc5 0.731
final eval loss 0.000 acc1 0.430 acc5 0.731
```
