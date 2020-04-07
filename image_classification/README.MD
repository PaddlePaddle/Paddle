# 高级api图像分类

## 数据集准备
在开始训练前，请确保已经下载解压好[ImageNet数据集](http://image-net.org/download)，并放在合适的目录下，准备好的数据集的目录结构如下所示：

```bash
/path/to/imagenet
    train
         n01440764
              xxx.jpg
              ...
         n01443537
              xxx.jpg
              ...
         ...
    val
         n01440764
              xxx.jpg
              ...
         n01443537
              xxx.jpg
              ...
         ...
```


## 训练
### 单卡训练
执行如下命令进行训练
```bash
python -u main.py --arch resnet50 /path/to/imagenet -d
```

### 多卡训练
执行如下命令进行训练
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch main.py --arch resnet50 -d /path/to/imagenet
```

## 预测

### 单卡预测
执行如下命令进行预测
```bash
python -u main.py --arch resnet50 -d --evaly-only /path/to/imagenet 
```

### 多卡预测
执行如下命令进行多卡预测
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch main.py --arch resnet50 --evaly-only /path/to/imagenet
```


## 参数说明


* **arch**: 要训练或预测的模型名称
* **device**: 训练使用的设备，'gpu'或'cpu'，默认值：'gpu'
* **dynamic**: 是否使用动态图模式训练
* **epoch**: 训练的轮数，默认值：120
* **learning-rate**: 学习率，默认值：0.1
* **batch-size**: 每张卡的batch size，默认值：64
* **output-dir**: 模型文件保存的文件夹，默认值：'output'
* **num-workers**: dataloader的进程数，默认值：4
* **resume**: 恢复训练的模型路径，默认值：None
* **eval-only**: 仅仅进行预测，默认值：False


## 模型

| 模型 | top1 acc | top5 acc |
| --- | --- | --- |
| ResNet50 | 76.28 | 93.04 | 
