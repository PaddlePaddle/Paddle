# 图像风格迁移

图像的风格迁移是卷积神经网络有趣的应用之一。那什么是风格迁移呢？下图第一列左边的图为相机拍摄的一张普通图片，右边的图为梵高的名画星空。那如何让左边的普通图片拥有星空的风格呢。神经网络的风格迁移就可以帮助你生成第二列的这样的图片。

<div align=center>
 <img src="images/markdown/img1.png" width = "600" height = "300"  />
</br>
 <img src="images/markdown/img2.png" width = "300" height = "300"  divalign=center />
<div align=left>


## 基本原理
风格迁移的目标就是使得生成图片的内容与内容图片（content image）尽可能相似。由于在计算机中，我们用一个一个像素点表示图片，所以两个图片的相似程度我们可以用每个像素点的欧式距离来表示。而两个图片的风格相似度，我们采用两个图片在卷积神经网络中相同的一层特征图的gram矩阵的欧式距离来表示。对于一个特征图gram矩阵的计算如下所示：

```python
# tensor shape is [1, c, h, w]
_, c, h, w = tensor.shape
tensor = fluid.layers.reshape(tensor, [c, h * w])
# gram matrix with shape: [c, c]
gram_matrix = fluid.layers.matmul(tensor, fluid.layers.transpose(tensor, [1, 0]))
```

最终风格迁移的问题转化为优化上述的两个欧式距离的问题。这里要注意的是，我们使用一个在imagenet上预训练好的模型vgg16，并且固定参数，优化器只更新输入的生成图像的值。

## 具体实现
接下来，使用代码一步一步来实现上述图片的风格迁移

```python
# 导入所需的模块
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from hapi.model import Model, Loss

from hapi.vision.models import vgg16
from hapi.vision.transforms import transforms
from paddle import fluid
from paddle.fluid.io import Dataset

import cv2
import copy

# 图像预处理函数，和tensor恢复到自然图像的函数
from .style_transfer import load_image, image_restore
```


```python
# 启动动态图模式
fluid.enable_dygraph()
```

```python
# 内容图像，用于风格迁移
content_path = './images/chicago_cropped.jpg'
# 风格图像
style_path = './images/Starry-Night-by-Vincent-Van-Gogh-painting.jpg'
```


```python
# 可视化两个图像
content = load_image(content_path)
style = load_image(style_path, shape=tuple(content.shape[-2:]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
ax1.imshow(image_restore(content))
ax2.imshow(image_restore(style))
```


![png](images/markdown/output_10_1.png)



```python
# 定义风格迁移模型，使用在imagenet上预训练好的vgg16作为基础模型
class StyleTransferModel(Model):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        # pretrained设置为true，会自动下载imagenet上的预训练权重并加载
        vgg = vgg16(pretrained=True)
        self.base_model = vgg.features
        for p in self.base_model.parameters():
            p.stop_gradient=True
        self.layers = {
                  '0': 'conv1_1',
                  '3': 'conv2_1',
                  '6': 'conv3_1',
                  '10': 'conv4_1',
                  '11': 'conv4_2',  ## content representation
                  '14': 'conv5_1'
                 }

    def forward(self, image):
        outputs = []
        for name, layer in self.base_model.named_sublayers():
            image = layer(image)
            if name in self.layers:
                outputs.append(image)
        return outputs
```


```python
# 定义风格迁移个损失函数
class StyleTransferLoss(Loss):
    def __init__(self, content_loss_weight=1, style_loss_weight=1e5, style_weights=[1.0, 0.8, 0.5, 0.3, 0.1]):
        super(StyleTransferLoss, self).__init__()
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.style_weights = style_weights

    def forward(self, outputs, labels):
        content_features = labels[-1]
        style_features = labels[:-1]

        # 计算图像内容相似度的loss
        content_loss = fluid.layers.mean((outputs[-2] - content_features)**2)

        # 计算风格相似度的loss
        style_loss = 0
        style_grams = [self.gram_matrix(feat) for feat in style_features ]
        style_weights = self.style_weights
        for i, weight in enumerate(style_weights):
            target_gram = self.gram_matrix(outputs[i])
            layer_loss = weight * fluid.layers.mean((target_gram - style_grams[i])**2)
            b, d, h, w = outputs[i].shape
            style_loss += layer_loss / (d * h * w)

        total_loss = self.content_loss_weight * content_loss + self.style_loss_weight * style_loss
        return total_loss

    def gram_matrix(self, A):
        if len(A.shape) == 4:
            batch_size, c, h, w = A.shape
            A = fluid.layers.reshape(A, (c, h*w))
        GA = fluid.layers.matmul(A, fluid.layers.transpose(A, [1, 0]))

        return GA
```


```python
# 创建模型
model = StyleTransferModel()
```


```python
# 创建损失函数
style_loss = StyleTransferLoss()
```


```python
# 使用内容图像初始化要生成的图像
target = Model.create_parameter(model, shape=content.shape)
target.set_value(content.numpy())
```


```python
# 创建优化器
optimizer = fluid.optimizer.Adam(parameter_list=[target], learning_rate=0.001)
```


```python
# 初始化高级api
model.prepare(optimizer, style_loss)
```


```python
# 使用内容图像和风格图像获取内容特征和风格特征
content_fetures = model.test(content)
style_features = model.test(style)
```


```python
# 将两个特征组合，作为损失函数的label传给模型
feats = style_features + [content_fetures[-2]]
```


```python
# 训练5000个step，每500个step画一下生成的图像查看效果
steps = 5000
for i in range(steps):
    outs = model.train(target, feats)

    if i % 500 == 0:
        print('iters:', i, 'loss:', outs[0])
        plt.imshow(image_restore(target))
        plt.show()
```

    iters: 0 loss: [8.829961e+09]



![png](images/markdown/output_20_1.png)


    iters: 500 loss: [3.728548e+08]



![png](images/markdown/output_20_3.png)


    iters: 1000 loss: [1.6327214e+08]



![png](images/markdown/output_20_5.png)


    iters: 1500 loss: [1.0806553e+08]



![png](images/markdown/output_20_7.png)


    iters: 2000 loss: [81069480.]



![png](images/markdown/output_20_9.png)


    iters: 2500 loss: [64284104.]



![png](images/markdown/output_20_11.png)


    iters: 3000 loss: [52580884.]



![png](images/markdown/output_20_13.png)


    iters: 3500 loss: [43825304.]



![png](images/markdown/output_20_15.png)


    iters: 4000 loss: [37048400.]



![png](images/markdown/output_20_17.png)


    iters: 4500 loss: [31719670.]



![png](images/markdown/output_20_19.png)



```python
# 风格迁移后的图像
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(image_restore(content))
ax2.imshow(image_restore(target))
ax3.imshow(image_restore(style))
```




![png](images/markdown/output_21_1.png)



## 总结
上述可运行的代码都在[style-transfer.ipynb](./style-transfer.ipynb)中， 同时我们提供了[style-transfer.py](./style-transfer.py)脚本，可以直接执行如下命令，实现图片的风格迁移：

```shell
python -u style-transfer.py --content-image /path/to/your-content-image --style-image /path/to/your-style-image --save-dir /path/to/your-output-dir
```

风格迁移生成的图像保存在```--save-dir```中。

## 参考

[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
