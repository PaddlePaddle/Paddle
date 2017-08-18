# Paddle Python模块设计

![DOT](http://api.paddlepaddle.org/graphviz?dot=https://raw.githubusercontent.com/reyoung/graphviz_dots/master/refactor/python_arch.dot)

这篇设计文档讨论Paddle重构后，如何设计Python模块。使面对用户的最高层是我们的`paddle.v2` API，而面对C++的是我们最底层重构后的`Operator`, `Scope`, etc。即上图中红色模块的部分。


经过大家的讨论和简单的书写，大家对于Python模块设计达成了初步共识。初步设计为

![design](https://raw.githubusercontent.com/reyoung/graphviz_dots/master/refactor/graph.png)

## Model

Model是训练神经网络需要的一些信息的汇总，其中记录:

1. 拓扑结构栈(栈是为了实现RNN或IfElseOp)
2. Scope栈
3. 参数初始化网络
4. 参数，以及参数和梯度的对应关系
5. 设备信息

其中，在Caffe2中与之对应的类型叫做`model_helper`，而Tensorflow中与之对应的类型是`session`。在春伟的PR中，他将类似的概念成为[`session`](https://github.com/PaddlePaddle/Paddle/pull/3566/files#diff-b6e4bb9095a126ed31ee1cdae03af483R9)。

需要注意的是，Model中并不实现某一层，譬如`fc_layer`是实现在全局函数中，而不是实现在Model类中。

同时，Model中的设备信息默认值是空。用户可以一开始设置设备，也可以在运行前设置设备。每一个model只绑定一个设备。如果一个Model已经被运行过(即调用过init_param()或run())，则Model的设备不能再变更。

`Model.run()`可以接受结束的`Expression`，这时候Paddle会运行这个模型中需要计算出这个Expression的子图。如果不指定，则运行全部Model。提取子图的过程可以被缓存(cache)住。提取子图算法描述如下:

```python
def extract_subnet_from(Net net, set<string> need_vars, int from=-1):
    end = max(len(net) - 1, from)
    subnet = []
    for (; end > 0; --end):
        op = net[end];
        if any of op.outputs in need_vars:
            subnet.append(op)
            need_vars.extend(op.inputs)

    return subnet.reverse()
```

用户可以创建任意多个Model，不同的Model可以分享同样的根 Scope。Paddle默认会有一个全局的Model。即

```python
g_model = Model()
```

## Model Methods

我们使用一些全局函数(或者全局类)修改Model。这些全局函数参数不同，实现不同，但都满足如下条件:

* 参数中带有`model`，即`fc(..., model=None)`。 如果用户不设置`model`的话，就是用默认全局的`g_model`。(这里实现了和v2 API的兼容。v2 API相当于使用了全局`model`的API)
* 接受其他Model Method的输入，即如果某一参数是其他层的输出，类型是Expression。
* 所有model method 返回 一个 Expression 或 Expression数组 或 None
* Model Method 修改Model中的拓扑结构，默认使用 `model.cur_net()` 获得栈顶网络。

大体上，Model Methods可以分为三类:

1. data_layer/parameter等不修改拓扑结构的model method
2. fc/conv等通过用户配置来修改拓扑结构的model method
3. sgd/backward等根据Model本身的情况来修改拓扑结构的model method


### 不修改拓扑结构的model method

为了同一Model Methods的输入输出类型，`data_layer`返回值也是一个Expression。一个DataLayer的实现方式为:

```python
def data_layer(name, shape, model=None):
  if model is None:
    model = g_model
  
  model.root_scope().new_var(name).get_tensor().resize(shape)
  return Expression(name=name, model=model, op_pos=-1)
```

而同理，对于模型的参数，也需要通过`parameter`讲其转换为Expression。方法如下:

```python
def parameter(name, dim, attr, model=None):
  if model is None:
    model = g_model
  
  # params are always created in root scope.
  if model.root_scope().find_var(name) is not None:
    # param has been created before, do not create it again.
    return Expression(name=name, model=model, op_pos=-1)
  model.root_scope().new_var(name).get_tensor()
  
  # This line could be changed by attr, not only uniform is supported.
  # just a demo here
  model.init_net.create_and_add_op("uniform_random", **attr)

  model.param_names.add(name)
  return Expression(name=name, model=model, op_pos=-1)
```

### 通过用户配置来修改拓扑结构的model method

通过用户配置来修改拓扑结构的Model Method需要注意的是:

1. 要使用`model.cur_net()`和`model.cur_scope()`来获取网络。
2. 如果要创建函数，需要使用`parameter`函数。

为什么一定要使用`cur_net()`，即栈顶的网络来创建新的拓扑结构呢？原因如下:

Paddle中的`NetOp`是一个op的数组。而RNNOp以及类似的IfElseOp都保存有一到多个内部网络。

```python
class NetOp():
  vector<Op*> ops_;

class RNNOp():
  NetOp step_net_;
```

而含有RNNOp的网络创建过程，如下图所示

![stack_of_op](https://raw.githubusercontent.com/reyoung/graphviz_dots/master/refactor/stack_of_op.png)

一个fc层函数的示例为:

```python
def fc(input, size, param_attr=None, bias_attr=False, act="sigmoid", model=None):
  model = default_model(model)
  dim = input.tensor().get_dims()
  w = parameter("w", dim=[dim[1], size], param_attr)
  tmp = model.cur_net().create_and_add_op("mul", X=input, Y=w)
  if bias_attr:
    b = parameter("b", dim=[size], bias_attr)
    tmp = model.cur_net().create_and_add_op("rowwise_add", X=tmp, Y=b)
  
  if act:
    tmp = model.cur_net().create_and_add_op(act, X=tmp)
  
  return Expression(tmp, model)
```

### 根据Model本身的情况来修改拓扑结构的model method

这里主要是指backward或者SGD等函数。他们虽然是对model的修改，但是不只添加一个或者有限个Op，而是根据model现有状况添加Op。譬如SGD method

```python
def sgd(learning_rate=1e-3, model=None):
  model = default_model(model)
  ret_val = []
  for param_name in model.param_grad_map:
    grad_name = model.param_grad_map[param_name]
    
    model.root_net().create_and_add_op("sgd", learning_rate=1e-3, X=param_name, Y=grad_name, Out=param_name)
    ret_val.append(Expression(param_name, model))
    
  return ret_val
```

而backward则主要调用C++中的`Backward`函数，并将生成的Network添加进`model.root_net()`



## Expression

Expression是统一后的Model Method的输入输出。春伟的PR中将这一概念命名为[`Var`](https://github.com/PaddlePaddle/Paddle/pull/3566/files#diff-c00711137ba1e93c609c893c649d302c)。经过讨论同步，这一概念需要有的操作有两个，他们是:

```python
class Expression():
  def name():
    return str('final variable name of this expression')
  
  def value():
    return numpy.ndarry('variable value after calculating')
```

在Expression中，保存了:

1. Model的指针
2. 计算出这个变量的Op在Model中的位置(如果是RNN，就是子图中的位置)。
3. 变量的名字

通过这些，我们可以使用`Expression.value()`获得某一个变量的计算结果。如果这一个变量没有记算过，(即Data更新了，但是网络没有计算)，则可以直接计算出来。

同时，`Model.run()`也可以接受若干个`Expression`进行计算。这在文档`Model`部分有所描述。
