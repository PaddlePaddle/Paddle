# Paddle大规模分布式训练设计

## 概览

## 常见的分布式训练架构

深度学习分布式训练的架构如图：

<img src="src/trainer.png"/>

为了完成一个深度学习的训练任务，集群中会运行多个trainer和parameter server，集群会把模型的参
数分布式的存储在多个parameter server上，trainer完成每个mini-batch数据训练之后会把梯度发送
给parameter server，parameter server将某个分片的模型参数和梯度执行整合和优化。然后trainer
从所有的parameter server下载模型参数并开始下一轮mini-batch的训练。

可以看到，可以进一步的优化以下方面：
1. 模型的参数是保存在parameter server进程的内存中的。在一个训练任务过程中任意一台
  parameter server不能异常退出，否则训练不能继续执行
1. 不能在一个训练任务中动态的增加Trainer个数或parameter个数
1. parameter server保存模型参数考虑多个备份防止单点故障
1. 为了使训练任务至少可以抵御“单点故障”（任意时刻只可能同时有一台服务器故障），模型参数的更新和分发
  需要保证原子性操作或满足事务性操作
1. 可以同时调度大量的训练任务和使用模型的应用在一个集群上
1. 支持训练任务的前置任务和后置任务，支持训练任务的定时调度和对在线流式数据的处理

## 模型参数数据备份
为了实现parameter server集群可以容忍单点故障，必须将每个模型参数的分片在集群中存储多个副本。虽然
也可以考虑使用校验和的技术减少副本大小，但为了整体系统的简单可靠，优先选择使用副本的方式。

<img src="src/replica.png"/>

上图显示了在2台parameter server中实现每个模型参数的分片均保存两个副本的状态。parameter 负责存储
所有参数分片副本并在etcd中同步每个副本的状态。每个分片的多个副本中同时只有一个处于"master"状态，
处于"master"状态的副本是当前活动的副本。当一台parameter server故障时，集群中剩下的parameter server
会重新选举出新的"master"副本并继续提供服务。

用户在启动parameter server是可以指定副本的个数(>=1)，副本越多容灾能力越强，越少性能越好。但通常不会
使用>3个的副本配置。

etcd中数据存储格式为：
1. pserver集群状态`[CLUSTER_CHROOT]/pserver_cluster_status`
  ```json
  {
    "cluster_status": "OK|UNHEALTHY|UNKNOWN"
    "reason": "",
    "nodes": [0,1,2,3]
  }
  ```

1. 每个pserver的状态: [CLUSTER_CHROOT]/pservers/[pserverid]
  ```json
  {
    "id": 0,
    "instance": "pserver1",
    "status": "up",
    "start_time": 1490184573.25,
    "sync": true,
  }
  ```
1. mini-batch计数器，记录此id对应的parameter server正在执行的mini batch id
  [CLUSTER_CHROOT]/pservers/[pserverid]/mini-batch-id
1. parameter分片信息: [CLUSTER_CHROOT]/pshards/[shardid]/[replicaid]
  比如上图显示的分片将生成下面的4个etcd路径：
  ```bash
    /pshards/0/0
    /pshards/0/1
    /pshards/1/0
    /pshards/1/1
  ```
  每个replica的信息如下：
  ```json
  {
    "id": 0,
    "shardid": 0,
    "created": 1490184573.25,
    "modified": 1490184573.25,
    "status": "master", # indicates the replica is in use
  }
  ```

## 数据一致性
存在多个副本数据的情况下就需要考虑，多个副本之间的数据一致性。如果使用数据强一致性，则在故障恢复时
可以获得一个完整的数据集，但每次更新模型参数的性能会下降，因为需要保证多个副本都完全更新之后才算更新
成功。如果使用异步同步（最终一致性），则在重新选举"master"副本时，可能得到的副本并没有完成数据同步。

## 故障恢复

## 动态扩容/缩容
