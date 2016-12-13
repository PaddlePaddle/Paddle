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

- `-i` 或 `--input` 给出输入数据所在路径；
- `-s` 或 `--size` 给出图片尺寸；
- `-c` 或 `--color` 标示图片是彩色图或灰度图

## 模型训练
在开始训练之前，我们需要先创建一个配置文件。下面我们给出了一个配置文件的示例（vgg_16_cifar.py）。**注意**，？？？

```python
from paddle.trainer_config_helpers import *
data_dir='data/cifar-out/batches/'
meta_path=data_dir+'batches.meta'
args = {'meta':meta_path, 'mean_img_size': 32,
        'img_size': 32, 'num_classes': 10,
        'use_jpeg': 1, 'color': "color"}
define_py_data_sources2(train_list=data_dir+"train.list",
                        test_list=data_dir+'test.list',
                        module='image_provider',
                        obj='processData',
                        args=args)
settings(
    batch_size = 128,
    learning_rate = 0.1 / 128.0,
    learning_method = MomentumOptimizer(0.9),
    regularization = L2Regularization(0.0005 * 128))

img = data_layer(name='image', size=3*32*32)
lbl = data_layer(name="label", size=10)
# small_vgg is predined in trainer_config_helpers.network
predict = small_vgg(input_image=img, num_channels=3)
outputs(classification_cost(input=predict, label=lbl))
```

在第一行中我们载入用于定义网络的函数。
```python
from paddle.trainer_config_helpers import *
```

之后`define_py_data_sources2`使用python数据接口进行定义，其中 `args`将在`image_provider.py`进行使用，后者负责将图片数据传递给Paddle
 - `meta`: the mean value of training set.
 - `mean_img_size`: the size of mean feature map.
 - `img_size`：输入图片的高度及宽度。
 - `num_classes`：分类的个数。
 - `use_jpeg`：处理过程中数据存储格式
 - `color`标示是否为彩色图片
 
 `settings`用于设置训练算法。在下面的例子中， it specifies learning rate as 0.1, but divided by batch size, and the weight decay is 0.0005 and multiplied by batch size.
 
 ```python
settings(
    batch_size = 128,
    learning_rate = 0.1 / 128.0,
    learning_method = MomentumOptimizer(0.9),
    regularization = L2Regularization(0.0005 * 128)
)
```

`small_vgg`定义了网络结构。这里我们使用了VGG卷积神经网络的一个小型版本。关于VGG卷积神经网络的描述可以参考：[http://www.robots.ox.ac.uk/~vgg/research/very_deep/](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)。
```python
# small_vgg is predined in trainer_config_helpers.network
predict = small_vgg(input_image=img, num_channels=3)
```
生成配置之后，我们就可以运行脚本train.sh来训练模型。请注意下面的脚本中假设该脚本放置是在路径`./demo/image_classification`下的。如果要从其它路径运行，你需要修改下面的脚本路径，以及配置文件中的相应内容。

```bash
config=vgg_16_cifar.py
output=./cifar_vgg_model
log=train.log

paddle train \
--config=$config \
--dot_period=10 \
--log_period=100 \
--test_all_data_in_one_period=1 \
--use_gpu=1 \
--save_dir=$output \
2>&1 | tee $log

python -m paddle.utils.plotcurve -i $log > plot.png
```

