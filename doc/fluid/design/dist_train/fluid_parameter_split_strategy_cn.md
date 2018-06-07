# Fluid 分布式训练模型参数切分策略详解
本篇文章将说明, 在使用 PaddlePaddle Fluid 进行基于 Parameter Server 的分布式训练时, 模型参数的分配方案设计, 并且举了一个如何使用这种切分方案的栗子:) ;

## 模型切分策略设计
### 参数切分原因

在模型设计时, 我们通常不会限制模型隔层使用的参数大小, 但当我们设计了如下的网络时:

![fluid_3_layer_network](src/fluid_3_layers_network.png)

fluid.input 层可能非常宽, 导致 w1, b1 参数纬度可能非常的大, 而 fluid.fc 层可能非常窄, 导致 w2, b2 参数纬度特别小, 如果只是简单的将模型分配到参数服务器上可能会导致每个参数服务器拿到的参数大小并不均匀, 负载较轻的参数服务器会等待负载较重的参数服务器, 所以针对参数大小不均匀的情况, 我们提供了参数切分功能;

### 参数切分方式

参数会在切分后变为参数块, 而在切分参数时, 如果参数切分的过细会导致参数服务器的计算效率不高, 但如果参数切分的不够均匀又无法达到我们上述的效果, 所以我们会先定一个最小的参数块大小: 8192, 并且按照如下方式计算需要切分数量:

```
# parameter_size: 参数大小
# MIN_PARAMETER_BLOCK_SIZE: 最小参数块大小, 8192
# parameter_server_count: 参数服务器总数
math.min(parameter_size / MIN_PARAMETER_BLOCK_SIZE, parameter_server_count)
```
在将参数切分为多个参数块后, 我们还需要对参数块进行打散, 均匀的分配到参数服务器上

### 参数分配方式
我们现在支持两种简单而有效的[Partition](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/transpiler/ps_dispatcher.py)方式: [Round Robin](https://en.wikipedia.org/wiki/Round-robin_scheduling) 和 [Hash](https://en.wikipedia.org/wiki/Hash_function);

在 Round Robin 模式中, 我们会 one-by-one 的将参数分配到 Server 上;

在 Hash 模式中, 我们会对参数块名称进行 Hash 操作然后对整体参数服务器数量取模, 得到具体的参数服务器Id;

```python
server_id = hash(block_str) % total
```
### 整体切分流程

经过参数分配后, 我们的参数切分和分配策略就结束了:

![fluid_parameter_slice_up](src/fluid_parameter_slice_up.png)


## 模型参数切分用例
### 分布式实现

PaddlePaddle Fluid分布式训练的具体实现方式可以参考 [Fluid Cluster Train](../../howto/cluster/fluid_cluster_train_cn.md)

### 参数详解
我们主要的参数策略实现在了 [Distribute Transpiler](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/transpiler/distribute_transpiler.py) 中, 我们可以在```transpile```方法中指定```slice_var_up=True```来开启模型参数切分, 并且可以使用```split_method=RoundRobin```来指定模型参数的分配方式, 示例代码如下:

```python
transpiler.transpile(
	trainer_id=trainer_id,
	slice_var_up=True,
	split_method=RoundRobin,
	pservers=pservers,
	trainers=trainers)
```
