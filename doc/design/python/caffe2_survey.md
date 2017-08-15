# Caffe2 Python实现调研

## Caffe2 Python端主要模块
![Python Concepts](http://api.paddlepaddle.org/graphviz?dot=https://gist.githubusercontent.com/reyoung/d9a9112730c514e0ee74aa5df8a8fa33/raw/fca60683d9b00eeb89f28e7065bd792b3b4ef4b4/caffe2_python.dot)

Caffe2在Python端构建拓扑结构的概念主要如上图所示:

1. Net: 一个神经网络拓扑
2. ModelHelper: 包含神经网络拓扑，和他的参数以及参数初始化网络
3. brew/optimizer/etc: 一些操作Model的帮助函数，也就是layer函数

用户可以直接调用这三层逻辑的任何一层。并且，对于复杂的神经网络，用户**必须**直接调用最底层的Net或者ModelHelper才能完成。

## 概念细节

### Net

[Net](https://github.com/caffe2/caffe2/blob/master/caffe2/python/core.py#L1298) 是Caffe2中最终生成神经网络拓扑结构的对象。使用Net进行训练的样例为(从官方[Toy Regression](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/Toy_Regression.ipynb)中摘录，将ipynb转换成Python代码):

```python
from caffe2.python import core, workspace

init_net = core.Net("init")

# The ground truth parameters.
W_gt = init_net.GivenTensorFill(
    [], "W_gt", shape=[1, 2], values=[2.0, 1.5])
B_gt = init_net.GivenTensorFill([], "B_gt", shape=[1], values=[0.5])
# Constant value ONE is used in weighted sum when updating parameters.
ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
# ITER is the iterator count.
ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0, dtype=core.DataType.INT32)

# For the parameters to be learned: we randomly initialize weight
# from [-1, 1] and init bias with 0.0.
W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
print('Created init net.')

train_net = core.Net("train")
# First, we generate random samples of X and create the ground truth.
X = train_net.GaussianFill([], "X", shape=[64, 2], mean=0.0, std=1.0, run_once=0)
Y_gt = X.FC([W_gt, B_gt], "Y_gt")
# We add Gaussian noise to the ground truth
noise = train_net.GaussianFill([], "noise", shape=[64, 1], mean=0.0, std=1.0, run_once=0)
Y_noise = Y_gt.Add(noise, "Y_noise")
# Note that we do not need to propagate the gradients back through Y_noise,
# so we mark StopGradient to notify the auto differentiating algorithm
# to ignore this path.
Y_noise = Y_noise.StopGradient([], "Y_noise")

# Now, for the normal linear regression prediction, this is all we need.
Y_pred = X.FC([W, B], "Y_pred")

# The loss function is computed by a squared L2 distance, and then averaged
# over all items in the minibatch.
dist = train_net.SquaredL2Distance([Y_noise, Y_pred], "dist")
loss = dist.AveragedLoss([], ["loss"])
gradient_map = train_net.AddGradientOperators([loss])

train_net.Iter(ITER, ITER)
LR = train_net.LearningRate(ITER, "LR", base_lr=-0.1,
                            policy="step", stepsize=20, gamma=0.9)

train_net.WeightedSum([W, ONE, gradient_map[W], LR], W)
train_net.WeightedSum([B, ONE, gradient_map[B], LR], B)

workspace.RunNetOnce(init_net)
workspace.CreateNet(train_net)
print("Before training, W is: {}".format(workspace.FetchBlob(W)))
print("Before training, B is: {}".format(workspace.FetchBlob(B)))
for i in xrange(1000):
    workspace.RunNet(train_net.Proto().name)

print("After training, W is: {}".format(workspace.FetchBlob(W)))
print("After training, B is: {}".format(workspace.FetchBlob(B)))

print("Ground truth W is: {}".format(workspace.FetchBlob(W_gt)))
print("Ground truth B is: {}".format(workspace.FetchBlob(B_gt)))
```

Net自身对于用户的接口主要是创建Operator，例如上一个代码片段中的:

* GivenTensorFill
* ConstantFill
* UniformFill
* GaussianFill
* FC
* WeightedSum

这些创建Operator的函数，并没有在Net对象中显示写明，而是运行时生成的函数。生成逻辑是:

* 调用Net.XXXOp函数，Python发现XXXOp函数并不存在
* Python将调用转发到[`Net.__getattr__`](https://github.com/caffe2/caffe2/blob/master/caffe2/python/core.py#L1951)函数中
* `Net.__getattr__`函数做了一些合法性检查，将调用转移到[`Net. _CreateAndAddToSelf`](https://github.com/caffe2/caffe2/blob/master/caffe2/python/core.py#L1919)函数中
* 在`_CreateAndAddToSelf`中，Op被创建并被添加到网络自身内部。
* 这一Op的输出Variable Name(Caffe2中是blob name)。加上一层薄封装，封装成[`BlobReference`](https://github.com/caffe2/caffe2/blob/master/caffe2/python/core.py#L1919)返回。

BlobReference 是每一个Caffe2 Op或者Layer都会返回的类型。他是一个简单的结构体，主要保存了:

* Blob的名字
* Blob所属的Net

作用是可以使用`Net.xxxOp`返回的变量继续创建Op。比如:

```
data = net.GivenTensorFill(...)
hidden = data.FC(...)
hidden = hidden.FC(...)
```

而`BlobReference.XXXOp`的调用原理和Net一致，都是运行时生成的函数。方法如下:

* Python将`XXXOp`转发到`BlobReference.__getattr__`中
* `BlobReference.__getattr__`调用`Net.XXXOp`，并将自己作为输入放入`XXXOp`的参数中。

### Model 与 Brew

为了管理参数初始化问题，Caffe2创建了Model类型。在Model中保存了:

* 网络拓扑(前向/反向/优化)
* 网络参数
    * 参数名
    * 参数名对应的梯度名
    * 哪些参数是Weight，哪些参数是bias
* 参数初始化网络
* 设备信息

Model并不直接提供`XXX_layer`的接口，而是提供了`create_parameter`, `get_param_info`等接口给层的开发者。

Caffe2中，对于层的封装，使用了一个约定的函数接口。他们约定第一个参数是`model`。而在这些层实现的函数中，或使用`Model`提供的接口(`create_parameter`等)，或直接操纵Model中的成员变量。

这些层函数大部分封装在了`brew`这个模块下。不过SGD，RNN分散在其他模块下。

用户的使用方法为:

```python
model = model_helper.ModelHelper(name="train_net")
fc1 = brew.fc(model, input, output, dim_in, dim_out, **kwargs)

```

当然，用户也可以(甚至在某些情况下必须)直接操作model类型的成员变量。这时，使用方法为:

```python
model = model_helper.ModelHelper(name="train_net")
# init your weight and bias as w and b
w = model.param_init_net.XavierFill(...)
b = model.param_init_net.ConstantFill(...)
fc1 = model.FC([input, w, b], output, **kwargs)
```

在Caffe2的教程中，也有两种明显不同的写法:

* 对于简单的前馈神经网络，[MNIST教程](https://github.com/caffe2/caffe2/blob/master/caffe2/python/tutorials/MNIST.ipynb)几乎全是使用`brew.xxx`来构造拓扑结构。
* 对于复杂的RNN网络，[CharLSTM](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/char_rnn.py)混合使用了`brew.xxx`和最底层的接口`Net.xxx`。


## 启发

1. Model可以管理多个拓扑结构，但构造Layer的函数，没必要也放到Model里面。可以放到`layer`模块下(`brew`不是一个好名字)。我们约定其中一个参数是`model`。但是可以将默认值设为一个全局model。

    ```python
    def fc_layer(input, size, act="sigmoid", model=None):
        if model is None:
           model = g_model  # use global model by default.
        ...
    ```
2. 可以由Net管理自己Op的内存。两个Net如果共享一些Op，那么就创建两个同样的Op给这两个不同的Net。Net析构的时候将自己的Op析构掉。因为Op本身主要是计算，占用内存不多(主要保存了Attribute)，故每个Net管理自己的Op可以简化内存管理。
