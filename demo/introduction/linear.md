# Paddle Python Trainer API使用说明

目前Paddle的PythonTrainerAPI处在一个测试阶段，很多设计都可以修改。包括接口。

## 开发周期

1.
使用Paddle的swig接口暴露出足够多支持Paddle训练的API，这个暴露级别最好在GradientMachine这个级别，可以方便用户自定义训练过程。
2. 在swig API的基础上，确定Python的用户接口是什么样子
3. 将swig API修改成C-API，进而可以多语言操控Paddle的训练

## 预期达到的效果

1. 用户可以完全使用Paddle Python的库完成训练。
  1. 训练过程中的信息，可以以强类型传递给Python端。
     * 正确率
     * cost
     * pass_id, batch_id
  2. 训练过程中的测试频率等工作，完全交由用户进行
     * 默认pass结束测试全部数据，但用户可以随意更改
  3. 用户可以非常自由的选择更新参数
     * 针对不同的数据，训练神经网络的不同部分。
       * 一组数据训练网络的左半边，另一组训练右半边
  4. 更方便的多目标学习

## Python端用户接口的封装

### 整体样例

下面以一个简单的线性回归作为用户接口的使用样例。这个线性回归是以 x和y为变量，回归一个 y=w*x+b
的方程。其中w和b预设为2和0.3。神经网络中的w和b都是学习出来的。

首先，先import一些包。Python的训练接口复用了目前Paddle的很多逻辑。

1. 网络配置复用的是trainer_config_helpers里面的配置
2. 数据传输复用的是PyDataProvider2的格式

`from py_paddle.trainer import *`这行import了Paddle Python端的训练函数。`import
py_paddle.swig_paddle as api`这行import了底层swig暴露的接口。

```python
from paddle.trainer_config_helpers import *
from paddle.trainer.PyDataProvider2 import *
from py_paddle.trainer import *
import py_paddle.swig_paddle as api
```

```python
@network(inputs={
    'x': dense_vector(1), 'y': dense_vector(1)
}, learning_rate=1e-3, batch_size=12)
def linear_network(x, y):
    y_predict = fc_layer(input=x, param_attr=ParamAttr(name='w'), size=1,
                         act=LinearActivation(), bias_attr=ParamAttr(name='b'))
    cost = regression_cost(input=y_predict, label=y)
    return cost
```

上面定义Paddle需要训练网络的网络结构。这个网络结构的定义使用了Python的函数。

`@network`是一个decorator，它将下面的函数变成Paddle的神经网络描述(protobuf)。其参数包括:

* inputs. 传输数据类型是一个字典，key是函数的参数名(这里就是x、y)，value是x，y对应的数据类型。这里数据类型都是dense的vector
  * 可用的数据类型参考PyDataProvider2的@provider的input_types类型
* 其余的参数是paddle的优化参数。参考`settings`

```python
help(settings)  # run this line to print document of settings method.
```

在linear_network里面定义的就是神经网络的计算图。返回值就是优化目标。

使用decorator `@network`，我们将这个函数封装成了一个Python类。进而，我们声明一个网络描述实例`linear`。

```python
linear = linear_network()
```

这个描述是实例里面包含了一些Paddle的计算图信息和网络输入顺序等等。下面几个block可以手动运行，展开输出。

```python
help(linear_network)  # run this line to print document of linear_network
```

```python
print linear.input_types()
```

```python
print linear.network_graph()  # Paddle neural network protobuf definition
```

```python
configs = {
    'w': 2,
    'b': 0.3
}
```

进而我们设置一下线性回归的参数。`y=w*x+b`， w和b设置为2和0.3。 这个dict被dataprovider使用。

```python
import random

@linear.provider()
def process(*args, **kwargs):
    for i in xrange(2000):
        x = random.random()
        yield {'x': [x], 'y': [configs['w'] * x + configs['b']]}

```

下一步是声明数据读取器(DataProvider)。其本身也是一个函数, `process`。

Paddle的PyDataProvider2的数据读取的主要想法是，用户只需要关注**从一个文件里**如何读取**一条数据**，然后按照一种数据格式yield出
去。其他batch组合，数据shuffle等工作Paddle完成。

声明这个DataProvider的过程，也是使用一个Decorator完成。注意这个decorator实际上是**linear实例的一个函数**。

这个函数的参数和PyDataProvider2一样，第一个是settings，第二个是filename。不过这里procees函数实际上没有使用任何参数，故pr
ocess中使用`*args, **kwargs`来接受任意参数。

返回值是使用yield返回。这里必须使用**字典**。

```python
help(process)
```

```python
runner = RunnerBuilder(network=linear).with_train_data(method=process).build()
```

下一步是构造一个Runner，Runner是Python Trainer API中的最基础数据类型。它具有的操作是

* 执行一个Pass。 run_one_pass。
* 增加一个Pass中的执行步骤，例如打印输出等等。

RunnerBulder是一个简单的Runner生成器。他负责将Paddle的训练流程插入到Runner的执行步骤中。

这里network传入linear对象，而训练数据的读取函数是process。调用build生成runner

关于Runner的具体说明参考其他文档，或者注释。

```python
learning_result = {
    'cost': [],
    'w': [],
    'b': []
}
```

我们声明一个learning_result字典，来保存训练过程中的数据，三个field分别保存每个pass后的误差，w值和b值。方便我们画图。

```python
with runner:
    while True:
        ctx = ContextWrapper(runner.run_one_pass())
        learning_result['cost'].append(ctx.cost())
        params = ctx.gradient_machine().getParameters()
        for param in params:
            learning_result[param.getName()].append(param.getBuf(api.PARAMETER_VALUE)[0])
        
        if abs(ctx.cost() - 0.0) < 1e-10:
            # end training.
            break
```

上面这个循环便是全部训练过程。

第一行with runner，是指我要使用runner这个类来进行训练了。在使用某一个runner前，必须使用with，来初始化一些数据。同时目前Paddle只
支持一个进程使用一个runner(Paddle的全局变量问题)。

每一个run_one_pass()会返回一个当前的context，使用context wrapper可以更好(类型安全)，更快(TODO
可以使用Cython优化)的访问Context。

```python
help(ctx)
```

这个训练过程中，我们不指定训练次数，而是指定当误差小于1e-10的时候，我们就退出。

同时，记录下每一个pass的w和b值。

之后我们便可以使用matplotlib画图。画图的方法不在赘述。是标准的matplotlib使用

```python
% matplotlib inline
import matplotlib.pyplot as plt

plt.plot("cost", data=learning_result)
plt.show()

```

```python
plt.plot("w", data=learning_result)
plt.show()

```

```python
plt.plot("b", data=learning_result)
plt.show()

```

至此，一个简单的Python Trainer API使用说明写完了。
