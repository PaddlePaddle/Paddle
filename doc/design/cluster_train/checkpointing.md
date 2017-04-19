# Paddle大规模分布式训练设计

## 概览
参考[这里](./README.md)

## 分布式训练架构

常见的深度学习分布式训练的架构如图：

<img src="src/trainer.png" width="500"/>

为了完成一个深度学习的训练任务，集群中会运行多个trainer和parameter server，每个trainer启动时，会先尝试从parameter server集群下载最新的参数，然后以mini-batch为单位读取训练数据集中的一部分数据(Data shard)。trainer会在训练过程中持续与parameter server通讯，上传计算出来的梯度以及下载最新的模型。

每个parameter server保存所有parameter的一个分片(Global model shard)，并负责接受所有trainer发送的梯度，完成SGD和优化算法，然后发送更新后的parameter到每个trainer。

这样，通过trainer和parameter server的分布式协作，可以完成神经网络的SGD方法的训练。Paddle可以同时支持同步SGD(synchronize SGD)和异步SGD(asynchronize SGD)。

在使用同步SGD训练神经网络时，Paddle使用同步屏障(barrier)，使梯度的提交和参数的更新按照顺序方式执行。在异步SGD中，则并不会等待所有trainer提交梯度才更新参数，这样极大的提高了计算的并行性：parameter server之间不相互依赖，并行的接收梯度和更新参数，parameter server也不会等待trainer全部都提交梯度之后才开始下一步，trainer之间也不会相互依赖，并行的执行模型的训练。可以看出，虽然异步SGD方式会提高参数更新并行度, 但是并不能保证参数同步更新，在任意时间某一台parameter server上保存的参数可能比另一台要更新，与同步SGD相比，梯度会有噪声。

在上面的分布式计算模型中，使用异步SGD比同步SGD可以一定程度的提供训练任务的容灾性。假设在某一时刻，一个trainer进程停止工作，其他的trainer仍然可以完成对部分数据的训练。

参考上面所描述的Paddle实现细节，可以进一步的优化以下方面：
1. 目前模型的参数是保存在parameter server进程的内存中的。在同步SGD或异步SGD训练过程中任意一台parameter server不能异常退出，否则参数丢失，训练不能继续执行。需要考虑每个模型分片(model shard)保存多个副本(replica)防止parameter server单点故障。
1. 不能在一个训练任务中动态的增加或减少Trainer个数或parameter个数（异步SGD是否可以增加Trainer?）
1. 在同步SGD训练过程中，需要保证参数更新满足事务性操作。即可能在更新参数过程中，存放这个参数的shard所在的服务器故障，就需要rollback并重新更新这个参数shard的其他存活副本。
1. 为了支持大量的训练任务和使用模型的应用在一个集群上，需要支持训练任务节点的伸缩。
1. 支持训练任务的前置任务和后置任务，支持训练任务的定时调度和对在线流式数据的处理

## 模型参数检查点(Checkpointing)
模型数据检查点的实现，可以有效的避免parameter server的单点或多点同时故障。模型参数检查点通过定期向磁盘上保存一份存储在parameter server内存中的模型数据的完整镜像，来保证训练过程可以从中间状态重新启动。在一个不可中断并缺少备份的训练任务中，可以通过阶段性的保存每个parameter server的数据快照(snapshot)到 ***分布式存储服务／分布式存储挂载点*** 达到容灾的目的，比如每隔10分钟或1小时保存最新的快照，并删除更早的快照。在出现单点故障时，只需要恢复这台节点，或者将这台节点迁移到另一个节点并启动即可恢复训练任务。

<img src="src/checkpointing.png" width="500"/>

### 快照保存的设计如下：

说明：

* parameter server在集群中启动后，自动挂载分布式存储目录，并把快照保存到这个目录下。
* 所有parameter server和trainer在etcd上注册自己的id节点为TTL节点`/ps/[id]`和`/trainer/[id]`，并保持心跳。
* ***注：trainer在故障恢复后，master会将失败的task重新分配给恢复的trainer执行。这样会引入更大的随机性。***
* ***注：parameter server在保存检查点时，利用了Linux内核的“写时复制”技术，在fork的进程中保存检查点，原进程可以继续接收trainer的梯度更新请求，而不影响检查点数据的保存。***
* ***注：每个parameter server的检查点各自独立保存，暂时不考虑多个parameter server同步的保存一个特定时间点的全局检查点，同样会引入随机性。***


检查点保存程序流程：

1. 如果满足条件""每个pass或每n个mini-batch"时，parameter server会`fork`自己，子进程中执行保存检查点任务，父进程继续工作。如果已经有子进程在进行保存检查点工作，则忽略。
2. parameter server生成一个UUID，向指定的目录中一个新的文件（文件名为此UUID）写入快照数据。在快照写入完成后，计算这个文件的MD5 sum。然后在etcd的`/checkpoints/[pserver_id]`中写入json内容：`{"uuid": [UUID], "md5", "MD5 sum", "timestamp": xxxx}`。
3. 删除磁盘目录中不是当前uuid的快照文件。
4. 关闭fork出来的进程。

这里需要用户额外注意，在您的实际环境中，训练任务的运行可能会占满trainer和parameter server之间的网络带宽，如果parameter server此时还需要通过网络访问分布式存储以保存快照，可能会造成网络拥塞，而出现阶段性的运行停滞。

### 从快照恢复

在parameter server第一次启动或任意时间parameter server故障后被Kubernetes重新启动，则需要回滚到上一个检查点：

  1. 从etcd中读取节点：`/checkpoints/[pserver_id]`获取最新的检查点的文件uuid
  1. 从磁盘文件中加载uuid文件名的检查点快照文件，并加载其中的参数
  1. 如果上面两步出现错误，则使用启动参数定义的初始化方法初始化参数
  1. 开始提供服务

## TODO List
### 推测执行/加速执行(TODO)
在异构集群中，如果存在某些trainer执行速度过慢会影响整体集群的速度（如图中Trainer 1），此时master将负责启动一个新的Trainer(Accelerate Trainer 2)，使用同样的训练数据block。哪个trainer先完成block的训练，则把另一个慢速的kill掉。

### 动态扩容/缩容
目前只考虑动态扩容trainer数量，可以减小系统复杂性。

## 术语
* model: 指深度学习训练之后得到的所有参数，使用这个神经网络可以完成对新数据的预测
* parameters: 神经网络中的参数，包括权重w和偏置b。一个神经网络的模型由大量的参数组成
* shard: 分片，通常指将一个整体拆分成多份的其中的一份。
* model shard: 将一个神经网络参数拆分成多份，每个shard分别存储在其中一台parameter server之上
* parameter block: 多个parameter block构成一个model shard
* 单点故障: 任意时刻只可能同时有一台服务器故障。由于集群中同时存在两台机器故障的概率极低（(平均故障率*平均故障修复时间)^2）只对特殊在线系统考虑两台以上同时故障的容灾。
