图像分类教程
==========

在本教程中，我们将带领你在CIFAR-10数据集上训练一个卷积神经网络，并使用它进行图像分类。如下图所示，卷积神经网络可以辨识图片中的主体，并给出分类结果。
<center>![Image Classification](./image_classification.png)</center>

## 数据准备
首先下载CIFAR-10数据集。下面是CIFAR-10数据集的官方网址：

<https://www.cs.toronto.edu/~kriz/cifar.html>

我们准备了一个脚本，可以用于从官方网站上下载CIFAR-10数据集，并将之转化为jpeg文件，存入我们为本文中的实验所设计的目录中。使用这个脚本前请确认已经安装了pillow及相关依赖模块。可以参照下面的命令进行安装和下载：

1. 安装pillow

```bash
sudo apt-get install libjpeg-dev
pip install pillow
```

2. 下载数据集

```bash
cd demo/image_classification/data/
sh download_cifar.sh
```

CIFAR-10数据集包含60000张32x32的彩色图片。图片分为10类，每个类包含6000张。其中50000张图片用于组成训练集，10000张组成测试集。

下图展示了所有的照片分类，并从每个分类中随机抽取了10张图片（为了和文件系统中的名称保持一致，我们保留了分类的英文名称）：
<center>![Image Classification](./cifar.png)</center>

脚本运行完成后，我们应当会得到一个名为cifar-out的文件夹，其下子文件夹的结构如下


```
train
---airplane
---automobile
---bird
---cat
---deer
---dog
---frog
---horse
---ship
---truck
test
---airplane
---automobile
---bird
---cat
---deer
---dog
---frog
---horse
---ship
---truck
```

cifar-out下包含`train`和`test`两个文件夹，其中分别包含了CIFAR-10中的训练数据和测试数据。我们可以通过如下命令进行预处理工作：

```
cd demo/image_classification/
sh preprocess.sh
```

其中`preprocess.sh` 调用 `./demo/image_classification/preprocess.py` 对图片进行预处理
```sh
export PYTHONPATH=$PYTHONPATH:../../
data_dir=./data/cifar-out
python preprocess.py -i $data_dir -s 32 -c 1
```

`./demo/image_classification/preprocess.py` 使用如下参数：

- `-i` 或 `--input` 指出输入数据所在路径；
- `-s` 或 `--size` 指出图片尺寸；
- `-c` 或 `--color` 指出图片是彩色图或灰度图
