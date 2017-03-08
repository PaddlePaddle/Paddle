# Paddle On Kubernetes Proposal
## 目的
  在易用性的角度对PaddlePaddle进行改进，使其更加好用，并且能够支持基于Kubernetes集群的分布式训练
## 功能特性
- 预先准备Docker Image

  首先用户需要准备一个Docker Image来准备训练数据。
  - 如何获取数据？

    可通过wget，HDFS client等等方式
  - 数据预处理

    PaddlePaddle可以提供一些工具来对训练数据进行一步预处理,例如数据分片等操作，用户将这些操作封装在一个Docker Image中。
- PaddlePaddle Client

  提供一个PaddlePaddle的二进制客户端，可通过命令行的方式启动分布式训练，值得注意的是，这个客户端是直接和Kubernetes的master进行通信，启动训练任务。

- Kubernetes Controller

  可以通过[ThirdPartyResource(TPR)](https://kubernetes.io/docs/user-guide/thirdpartyresources/)的方式，可以自定义PaddlePaddle的资源类型，来做这些事情：
  - 运行数据预处理的Job，并将数据存储在分布式存储上（ClusterFS or Ceph）
  - 待上一步执行成功后，执行pserver以及trainer的Job。

  目前预想的YAML文件可能是这个样子：
  ```YAML
  apiVersion: bash/v1
  kind:PaddlePaddle
  ...
  spec:
    parallelism: 3
    completions: 3
    prepareData:
    - name: prepare_data
      image: [your repo]/prepare_data:latest
    containers:
    - name: trainer
      image: [your repo]/paddle_train:latest
      ...
  ```
  之所以没有采用Kubernetes中[Init Container](https://kubernetes.io/docs/concepts/abstractions/init-containers)的机制是因为目前Init Container会在每个Pod启动之前启动一次，而不是整个训练任务启动之前执行。虽然可以通过文件锁等方式避免并发执行，但总是不够直接。
 
- PaddlePaddle 的容错处理
  - 改造PaddlePaddle,保证在任意时间，pserver或者trainer可以挂掉并且不影响最终结果。(待补充)
