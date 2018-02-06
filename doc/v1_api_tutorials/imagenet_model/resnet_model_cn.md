# Model Zoo - ImageNet #

[ImageNet](http://www.image-net.org/) 是通用物体分类领域一个众所周知的数据库。本教程提供了一个用于ImageNet上的卷积分类网络模型。

## ResNet 介绍

论文 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) 中提出的ResNet网络结构在2015年ImageNet大规模视觉识别竞赛(ILSVRC 2015)的分类任务中赢得了第一名。他们提出残差学习的框架来简化网络的训练，所构建网络结构的的深度比之前使用的网络有大幅度的提高。下图展示的是基于残差的连接方式。左图构造网络模块的方式被用于34层的网络中，而右图的瓶颈连接模块用于50层，101层和152层的网络结构中。

<center>![resnet_block](./resnet_block.jpg)</center>
<center>图 1. ResNet 网络模块</center>

本教程中我们给出了三个ResNet模型，这些模型都是由原作者提供的模型<https://github.com/KaimingHe/deep-residual-networks>转换过来的。我们使用PaddlePaddle在ILSVRC的验证集共50,000幅图像上测试了模型的分类错误率，其中输入图像的颜色通道顺序为**BGR**，保持宽高比缩放到短边为256，只截取中心方形的图像区域。分类错误率和模型大小由下表给出。
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">
<colgroup>
<col  class="left" />
<col  class="left" />
<col  class="left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="left">ResNet</th>
<th scope="col" class="left">Top-1</th>
<th scope="col" class="left">Model Size</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">ResNet-50</td>
<td class="left">24.9%</td>
<td class="left">99M</td>
</tr>
<tr>
<td class="left">ResNet-101</td>
<td class="left">23.7%</td>
<td class="left">173M</td>
</tr>
<tr>
<td class="left">ResNet-152</td>
<td class="left">23.2%</td>
<td class="left">234M</td>
</tr>
</tbody>

</table></center>
<br>

## ResNet 模型

50层，101层和152层的网络配置文件可参照```demo/model_zoo/resnet/resnet.py```。你也可以通过在命令行参数中增加一个参数如```--config_args=layer_num=50```来指定网络层的数目。

### 网络可视化

你可以通过执行下面的命令来得到ResNet网络的结构可视化图。该脚本会生成一个dot文件，然后可以转换为图片。需要安装graphviz来转换dot文件为图片。

```
cd demo/model_zoo/resnet
./net_diagram.sh
```

### 模型下载

```
cd demo/model_zoo/resnet
./get_model.sh
```
你可以执行上述命令来下载所有的模型和均值文件，如果下载成功，这些文件将会被保存在```demo/model_zoo/resnet/model```路径下。

```
mean_meta_224  resnet_101  resnet_152  resnet_50
```
   * resnet_50: 50层网络模型。
   * resnet_101: 101层网络模型。
   * resnet_152: 152层网络模型。
   * mean\_meta\_224: 均值图像文件，图像大小为3 x 224 x 224，颜色通道顺序为**BGR**。你也可以使用这三个值: 103.939, 116.779, 123.68。

### 参数信息

* **卷积层权重**

  由于每个卷积层后面连接的是batch normalization层，因此该层中没有偏置(bias)参数，并且只有一个权重。
  形状: `(Co, ky, kx, Ci)`
   * Co: 输出特征图的通道数目
   * ky: 滤波器核在垂直方向上的尺寸
   * kx: 滤波器核在水平方向上的尺寸
   * Ci: 输入特征图的通道数目

  二维矩阵: (Co * ky * kx, Ci), 行优先次序存储。

* **全连接层权重**

  二维矩阵: (输入层尺寸, 本层尺寸), 行优先次序存储。

* **[Batch Normalization](<http://arxiv.org/abs/1502.03167>) 层权重**

本层有四个参数，实际上只有.w0和.wbias是需要学习的参数，另外两个分别是滑动均值和方差。在测试阶段它们将会被加载到模型中。下表展示了batch normalization层的参数。
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">
<colgroup>
<col  class="left" />
<col  class="left" />
<col  class="left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="left">参数名</th>
<th scope="col" class="left">尺寸</th>
<th scope="col" class="left">含义</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">_res2_1_branch1_bn.w0</td>
<td class="left">256</td>
<td class="left">gamma, 缩放参数</td>
</tr>
<tr>
<td class="left">_res2_1_branch1_bn.w1</td>
<td class="left">256</td>
<td class="left">特征图均值</td>
</tr>
<tr>
<td class="left">_res2_1_branch1_bn.w2</td>
<td class="left">256</td>
<td class="left">特征图方差</td>
</tr>
<tr>
<td class="left">_res2_1_branch1_bn.wbias</td>
<td class="left">256</td>
<td class="left">beta, 偏置参数</td>
</tr>
</tbody>

</table></center>
<br>

### 参数读取

使用者可以使用下面的Python脚本来读取参数值:

```
import sys
import numpy as np

def load(file_name):
    with open(file_name, 'rb') as f:
        f.read(16) # skip header for float type.
        return np.fromfile(f, dtype=np.float32)

if __name__=='__main__':
    weight = load(sys.argv[1])
```

或者直接使用下面的shell命令:

```
od -j 16 -f _res2_1_branch1_bn.w0
```

## 特征提取

我们提供了C++和Python接口来提取特征。下面的例子使用了`demo/model_zoo/resnet/example`中的数据，详细地展示了整个特征提取的过程。

### C++接口

首先，在配置文件中的`define_py_data_sources2`里指定图像数据列表，具体请参照示例`demo/model_zoo/resnet/resnet.py`。

```
    train_list = 'train.list' if not is_test else None
    # mean.meta is mean file of ImageNet dataset.
    # mean.meta size : 3 x 224 x 224.
    # If you use three mean value, set like:
    # "mean_value:103.939,116.779,123.68;"
    args={
        'mean_meta': "model/mean_meta_224/mean.meta",
        'image_size': 224, 'crop_size': 224,
        'color': True,'swap_channel:': [2, 1, 0]}
    define_py_data_sources2(train_list,
                           'example/test.list',
                           module="example.image_list_provider",
                           obj="processData",
                           args=args)
```

第二步，在`resnet.py`文件中指定要提取特征的网络层的名字。例如，

```
Outputs("res5_3_branch2c_conv", "res5_3_branch2c_bn")
```

第三步，在`extract_fea_c++.sh`文件中指定模型路径和输出的目录，然后执行下面的命令。

```
cd demo/model_zoo/resnet
./extract_fea_c++.sh
```

如果执行成功，特征将会存到`fea_output/rank-00000`文件中，如下所示。同时你可以使用`load_feature.py`文件中的`load_feature_c`接口来加载该文件。

```
-0.115318 -0.108358 ... -0.087884;-1.27664 ... -1.11516 -2.59123;
-0.126383 -0.116248 ... -0.00534909;-1.42593 ... -1.04501 -1.40769;
```

* 每行存储的是一个样本的特征。其中，第一行存的是图像`example/dog.jpg`的特征，第二行存的是图像`example/cat.jpg`的特征。
* 不同层的特征由分号`;`隔开，并且它们的顺序与`Outputs()`中指定的层顺序一致。这里，左边是`res5_3_branch2c_conv`层的特征，右边是`res5_3_branch2c_bn`层特征。

### Python接口

示例`demo/model_zoo/resnet/classify.py`中展示了如何使用Python来提取特征。下面的例子同样使用了`./example/test.list`中的数据。执行的命令如下：

```
cd demo/model_zoo/resnet
./extract_fea_py.sh
```

extract_fea_py.sh:

```
python classify.py \
     --job=extract \
     --conf=resnet.py\
     --use_gpu=1 \
     --mean=model/mean_meta_224/mean.meta \
     --model=model/resnet_50 \
     --data=./example/test.list \
     --output_layer="res5_3_branch2c_conv,res5_3_branch2c_bn" \
     --output_dir=features

```
* \--job=extract:              指定工作模式来提取特征。
* \--conf=resnet.py:           网络配置文件。
* \--use_gpu=1:                指定是否使用GPU。
* \--model=model/resnet_50:    模型路径。
* \--data=./example/test.list: 数据列表。
* \--output_layer="xxx,xxx":   指定提取特征的层。
* \--output_dir=features:      输出目录。

如果运行成功，你将会看到特征存储在`features/batch_0`文件中，该文件是由cPickle产生的。你可以使用`load_feature.py`中的`load_feature_py`接口来打开该文件，它将返回如下的字典：

```
{
'cat.jpg': {'res5_3_branch2c_conv': array([[-0.12638293, -0.116248  , -0.11883899, ..., -0.00895038, 0.01994277, -0.00534909]], dtype=float32), 'res5_3_branch2c_bn': array([[-1.42593431, -1.28918779, -1.32414699, ..., -1.45933616, -1.04501402, -1.40769434]], dtype=float32)},
'dog.jpg': {'res5_3_branch2c_conv': array([[-0.11531784, -0.10835785, -0.08809858, ...,0.0055237, 0.01505112, -0.08788397]], dtype=float32), 'res5_3_branch2c_bn': array([[-1.27663755, -1.18272924, -0.90937918, ..., -1.25178063, -1.11515927, -2.59122872]], dtype=float32)}
}
```

仔细观察，这些特征值与上述使用C++接口提取的结果是一致的。

## 预测

`classify.py`文件也可以用于对样本进行预测。我们提供了一个示例脚本`predict.sh`，它使用50层的ResNet模型来对`example/test.list`中的数据进行预测。

```
cd demo/model_zoo/resnet
./predict.sh
```

predict.sh调用了`classify.py`:

```
python classify.py \
     --job=predict \
     --conf=resnet.py\
     --multi_crop \
     --model=model/resnet_50 \
     --use_gpu=1 \
     --data=./example/test.list
```
* \--job=extract:              指定工作模型进行预测。
* \--conf=resnet.py:           网络配置文件。network configure.
* \--multi_crop:               使用10个裁剪图像块，预测概率取平均。
* \--use_gpu=1:                指定是否使用GPU。
* \--model=model/resnet_50:    模型路径。
* \--data=./example/test.list: 数据列表。

如果运行成功，你将会看到如下结果，其中156和285是这些图像的分类标签。

```
Label of example/dog.jpg is: 156
Label of example/cat.jpg is: 282
```
