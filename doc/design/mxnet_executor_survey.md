## 整体思路

### 执行流程：

- 用户通过Symbol来描述网络拓扑结构

- Symbol书写完毕后，会bind到一个Executor上

- Executor里面会完成Graph的构建，包括插入backward operator/copy operator等；同时完成InferShape/InferType等，并分配内存(这里需要注意的是，当输入数据的大小发生变化时，需要重新bind得到一个新的Executor)

- Executor有一个RunOps方法，在这里依次把operator的操作push到Engine中

- push到Engine的操作是异步执行的，Engine会对push进来的操作进行依赖分析；满足依赖的操作则发起执行，可以做一定程度的并行


### Python接口设计

Python端暴露的核心概念是Symbol和Module

- Symbol

Symbol是用户用来描述网络拓扑结构的，可以参考如下一个例子：

```python
def get_symbol(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp
```


- Module

Module是用来执行Symbol的，在里面会完成绑定Symbol得到Executor，参数初始化，数据读取，forward/backward计算，参数optimizer等过程。

神经网络计算图中的数据有三大类，一类是输入数据，一类是参数，一类是计算的中间结果。

**值得注意的是Mxnet对parameter的处理，有关初始化/加载/存储/更新操作都不是Operator！数据的读取操作也不是Operator！只有forward和backward的过程，才使用Operator来描述**


Mxnet对输入数据，参数的操作接口暴露是命令式的，非常清晰，容易理解

- 输入数据加载，参数初始化/加载/存储，本质上是对变量的set/load/save操作，直接操作比使用Operator更加简便
- 参数的更新，本质上是对变量读取之后，进行计算，然后再assign的操作，使用的是同一个内存；如果作为Operator加入到Graph中，会带来环，不利于优化

Mxnet对于计算中间结果的处理是符号式的，使用基于Operator组成的Graph来描述

- Forward/Backward的计算过程，基于Graph可以做一定程度的优化



## 数据并行，单机多卡


1. 构造DataParallelExecutorGroup

在Python端，Module包含一个成员变量`executor_group`，Module会把用户配置的Symbol绑定
到`executor_group`中，`executor_group`会根据用户传入的设备个数，构造出对应的个数的executor。具体构造实现如下：

```python
 self._exec_group = DataParallelExecutorGroup(self._symbol, 
                                              self._context,...)
```

这里的`context`是一个list，可以接收多个设备的id，比如

```python
context=[mx.gpu(0), mx.gpu(2)]
```
用来做各个设备上的数据并行。

2. 初始化kv-store

Mxnet的参数存储在一个kvstore中，在最开始模型所需要的参数进行初始化时，就是对kvstore的初始化，同时设置对应的optimizer方法


3. forward/backward

这里，每个设备上的executor会分别执行forward/backward


4. update

Module中提供了update方法，来负责优化参数，更新存储在kvstore上的参数：

```
def update(self):
    self.optimizer_initialized

    self._params_dirty = True
    if self._update_on_kvstore:
        _update_params_on_kvstore(self._exec_group.param_arrays,
                                      self._exec_group.grad_arrays,
                                      self._kvstore)
    else:
        _update_params(self._exec_group.param_arrays,
                       self._exec_group.grad_arrays,
                       updater=self._updater,
                       num_device=len(self._context),
                       kvstore=self._kvstore)
```


kvstore和dependency engine的结合，可以使得Mxnet的参数更新过程被很好的调度，计算与通信相互掩盖，达到较好的加速比。

## 模型并行，单机上跨设备


1. python端需要指定一个 group2ctx的参数，里面是一个key-value的map，value是设备id

 
```python
ngpu = 2
# A simple two GPU placement plan
group2ctx = {'embed': mx.gpu(0),
             'decode': mx.gpu(ngpu - 1)}
```


2. 在配置模型并行的拓扑结构时，需要构造Symbol，并且给每层指定一个key，这个key就是上面map中的key

```python
with mx.AttrScope(ctx_group='embed'):
        embed_weight=mx.sym.Variable("embed_weight")

with mx.AttrScope(ctx_group='decode'):
        cls_weight = mx.sym.Variable("cls_weight")
        cls_bias = mx.sym.Variable("cls_bias")
```

3. 把拓扑结构与group2ctx绑定起来，构造一个C++端的executor


```python
rnn_exec = rnn_sym.bind(..., group2ctx=group2ctx)
```


4. C++端的executor构造一个图，在NNVM里面专门有一个place_device的pass，来对Graph中的每个节点进行遍历，设置device信息；如果发现跨设备，则插入copy operator

调用关系图如下：
```
GraphExecutor::InitGraph  --> GraphExecutor::AssignContext  --> nnvm::pass::PlaceDevice
```
