## Paddle compile time vs runtime

PaddlePaddle是一个基于graph of operator的神经网络计算框架，正常的计算流程为：

1. 描述一个计算图
1. 分析、优化、修改计算图
1. 分配运行资源并执行

因此需要区分paddle编译时和运行时的状态。

关于编译时和运行时的划分，目前有两种想法：

1. 用一个Proto message: [VarDesc](https://github.com/PaddlePaddle/Paddle/pull/3693/files#diff-f4d158d316b6354763633a7f1930e072R41)来描述tensor。
  - 构件图的时候所有的数据以这种形式存在，InferShape直接修改这个VarDesc结构。
  - 运行的时候，通过VarDesc在Scope中创建对应的Variable。
1. 直接在某个Scope中创建Variable中的Tensor。
  - 构建图的时候就在Scope中创建相关Tensor，InferShape需要接受Scope作为参数，并修改Scope中的tensor的shape属性。
  - 运行的时候直接接受同样的Scope作为参数，并为所有Tensor分配内存。

### InferShape调用时机
目前的想法中，InferShape因为涉及到编译时和运行时两个阶段，对设计有较大影响。

一个问题是: InferShape调用几次，什么时机调用？

1. 编译时调用。为了实现边配置网络边检查shape的正确性，在创建Operator之后，必须做一次InferShape，这是没有争议的。
2. 运行是调用。运行时调用InferShape有以下几种原因：
   - 编译时某些维度没有设置，比如batch_size，只有在运行时设置了数据的维度才能正确Infer出来。
   - 编译完之后，用户手动修改了某些参数的维度，如果不做check，将无法发现错误。
   - 有些维度，必须在看到真实数据之后才能确定，例如condition op(if else while)等，他们的分支选择之后的数据size，必须知道真实数据之后才能确定。

**结论是**：InferShape需要在编译时和运行时都被调用，编译时主要用于check size的配置是否正确，运行时一方面要check size是否正确，还要根据真实数据来做一些resize.


下面分析两种情况的优缺点：

### 1. VarDesc作为编译时，Scope/Variable作为运行时
#### 实现方法
VarDesc创建的时候无需Scope，只需要在一个全局map中保存VarDesc，InferShape需要修改对应的VarDesc即可。

#### 优点
1. 切换Scope简单，只需在Op Run的时候传入一个新的Scope，框架根据全局的VarDesc Map在其中创建对应的Var即可运行。
2. 用VarDesc存储元信息，方便做图的优化。
3. InferShape就可以不需要传入Scope这个参数，因为修改的VarDesc都存在于全局的map中
4. 在分布式场景下，需要将图序列化之后发送给别的机器执行，这个终归是需要将Variable的相关属性也序列化的。这个点带来一个好处是，云端执行任务是可控的，用户发过来的是一个序列化的图，而不是一个脚本的源代码，有利于数据安全。

#### 缺点
1. InferShape实现复杂，编译时InferShape是基于VarDesc，但是运行时也同样需要做InferShape和resize()，因为
	a. 运行时size可能会被用户修改.
	b. Op实现也要求运行时需要做InferShape(例如RNN)

这两种InferShape一个基于VarDesc，一个基于Scope中的Variable进行，代码上会有一定的重复(需要想清楚有多复杂，一种想法是封装在InferContext和Executioncontext中)。

### 2. 不带内存的Tensor作为编译时，带内存的Tensor作为运行时。
#### 实现方法
Var必须创建在某个已经存在的Scope中，切换Scope之前，需要clone某个创建过Var的Scope

#### 优点
1. InferShape只需要实现一遍，配置时和运行时都调用同样的函数即可

#### 缺点
1. Clone的实现可能并不简单，比如多种设备类型之间内存如何同步(Scope for CPU vs Scope for GPU)
1. 实现Graph的序列化，最终还是需要一个类似VarDesc的角色。
