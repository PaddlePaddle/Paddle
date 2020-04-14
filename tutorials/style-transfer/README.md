# 图像风格迁移


图像的风格迁移是卷积神经网络有趣的应用之一。那什么是风格迁移呢？下图第一列左边的图为相机拍摄的一张普通图片，右边的图为梵高的名画星空。那如何让左边的普通图片拥有星空的风格呢。神经网络的风格迁移就可以帮助你生成第二列的这样的图片。


<div align=center>
 <img src="images/markdown/img1.png" width = "600" height = "300"  />
</br>
 <img src="images/markdown/img2.png" width = "300" height = "300" />

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

## 风格迁移
执行如下命令，就可以进行风格迁移。生成的图像会保存在```--save-dir```中。
```python
python -u style-transfer.py --content-image /path/to/your-content-image --style-image /path/to/your-content-image --save-dir /path/to/your-output-dir
```

具体的生成过程也可以参考[style-transfer.ipynb](./hapi-style-transfer.ipynb)


## 参考文献

[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
