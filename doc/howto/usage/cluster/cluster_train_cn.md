# PaddlePaddle分布式训练

* [概述](#概述)
* [环境准备](#环境准备)
* [启动参数说明](#启动参数说明)
  * [启动参数服务器](#启动参数服务器)
  * [启动计算节点](#启动计算节点)
  * [准备数据集](#准备数据集)
  * [准备训练程序](#准备训练程序)
* [使用分布式计算平台或工具](#使用分布式计算平台或工具)
  * [使用Fabric启动集群作业](#使用fabric启动集群作业)
     * [准备一个Linux集群](#准备一个linux集群)
     * [启动集群作业](#启动集群作业)
     * [终止集群作业](#终止集群作业)
     * [检查集群训练结果](#检查集群训练结果)
     * [检查模型输出](#检查模型输出)
  * [在OpenMPI集群中提交训练作业](#在openmpi集群中提交训练作业)
     * [准备OpenMPI集群](#准备OpenMPI集群)
     * [启动集群作业](#启动集群作业-1)
  * [在Kubernetes集群中提交训练作业](#在kubernetes集群中提交训练作业)

## 概述
本文将介绍如何使用PaddlePaddle在不同的集群框架下完成分布式训练。分布式训练架构如下图所示：

<img src="https://user-images.githubusercontent.com/13348433/31772175-5f419eca-b511-11e7-9db7-5231fe3d9ccb.png" width="500">

- 数据分片（Data shard): 用于训练神经网络的数据，被切分成多个部分，每个部分分别给每个trainer使用。
- 计算节点（Trainer）: 每个trainer启动后读取切分好的一部分数据，开始神经网络的“前馈”和“后馈”计算，并和参数服务器通信。在完成一定量数据的训练后，上传计算得出的梯度（gradients），然后下载优化更新后的神经网络参数（parameters）。
- 参数服务器（Parameter server）:每个参数服务器只保存整个神经网络所有参数的一部分。参数服务器接收从计算节点上传的梯度，并完成参数优化更新，再将更新后的参数下发到每个计算节点。

这样，通过计算节点和参数服务器的分布式协作，可以完成神经网络的SGD方法的训练。PaddlePaddle可以同时支持同步随机梯度下降（SGD）和异步随机梯度下降。

在使用同步SGD训练神经网络时，PaddlePaddle使用同步屏障（barrier），使梯度的提交和参数的更新按照顺序方式执行。在异步SGD中，则并不会等待所有trainer提交梯度才更新参数，这样极大地提高了计算的并行性：参数服务器之间不相互依赖，并行地接收梯度和更新参数，参数服务器也不会等待计算节点全部都提交梯度之后才开始下一步，计算节点之间也不会相互依赖，并行地执行模型的训练。可以看出，虽然异步SGD方式会提高参数更新并行度, 但是并不能保证参数同步更新，在任意时间某一台参数服务器上保存的参数可能比另一台要更新，与同步SGD相比，梯度会有噪声。

## 环境准备

1. 准备您的计算集群。计算集群通常由一组（几台到几千台规模）的Linux服务器组成。服务器之间可以通过局域网（LAN）联通，每台服务器具有集群中唯一的IP地址（或者可被DNS解析的主机名）。集群中的每台计算机通常被成为一个“节点”。
1. 我们需要在集群的所有节点上安装 PaddlePaddle。 如果要启用GPU，还需要在节点上安装对应的GPU驱动以及CUDA。PaddlePaddle的安装可以参考[build_and_install](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/getstarted/build_and_install)的多种安装方式。我们推荐使用[Docker](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/getstarted/build_and_install/docker_install_cn.rst)安装方式来快速安装PaddlePaddle。

安装完成之后，执行下面的命令可以查看已经安装的版本（docker安装方式可以进入docker容器执行：`docker run -it paddlepaddle/paddle:[tag] /bin/bash`）：
```bash
$ paddle version
PaddlePaddle 0.10.0, compiled with
    with_avx: ON
    with_gpu: OFF
    with_double: OFF
    with_python: ON
    with_rdma: OFF
    with_timer: OFF
```

下面以`doc/howto/usage/cluster/src/word2vec`中的代码作为实例，介绍使用PaddlePaddle v2 API完成分布式训练。

## 启动参数说明
### 启动参数服务器
执行以下的命令启动一个参数服务器并等待和计算节点的数据交互
```bash
$ paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1
```

如果希望可以在后台运行pserver程序，并保存输出到一个日志文件，可以运行：
```bash
$ stdbuf -oL /usr/bin/nohup paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1 &> pserver.log
```

| 参数  | 是否必选 | 默认值 | 说明 |
| ------------- | ------------- | ------------- | ------------- |
| port  | 必选 | 7164 | pserver监听的起始端口，根据ports_num决定<br>总端口个数，从起始端口监听多个端口用于通信  |
| ports_num  | 必选 | 1 | 监听的端口个数  |
| ports_num_for_sparse  | 必选 | 1 | 用于稀疏类型参数通信的端口个数  |
| num_gradient_servers  | 必选 | 1 | 当前训练任务pserver总数 |

### 启动计算节点
执行以下命令启动使用python编写的trainer程序（文件名为任意文件名，如train.py）
```bash
$ python train.py
```

trainer需要和pserver保持网络联通以完成训练。trainer启动需要传入端口、pserver地址等参数使trainer可以正确连接到pserver。这些参数可以通过环境变量（https://zh.wikipedia.org/wiki/环境变量 ）或编写程序时`paddle.init()`中传入参数。如果同时使用`paddle.init()`参数和环境变量，将会优先使用`paddle.init()`中传入的参数。

使用环境变量：

```bash
export PADDLE_INIT_USE_GPU=False
export PADDLE_INIT_TRAINER_COUNT=1
export PADDLE_INIT_PORT=7164
export PADDLE_INIT_PORTS_NUM=1
export PADDLE_INIT_PORTS_NUM_FOR_SPARSE=1
export PADDLE_INIT_NUM_GRADIENT_SERVERS=1
export PADDLE_INIT_TRAINER_ID=0
export PADDLE_INIT_PSERVERS=127.0.0.1
```

使用参数：

```python
paddle.init(
        use_gpu=False,
        trainer_count=1,
        port=7164,
        ports_num=1,
        ports_num_for_sparse=1,
        num_gradient_servers=1,
        trainer_id=0,
        pservers="127.0.0.1")
```

| 参数  | 是否必选 | 默认 | 说明 |
| ------------- | ------------- | ------------- | ------------- |
| use_gpu  | 可选 | False | 是否启用GPU训练 |
| trainer_count  | 必选 | 1 | 当前训练任务trainer总个数 |
| port  | 必选 | 7164 | 连接到pserver的端口  |
| ports_num  | 必选 | 1 | 连接到pserver的端口个数  |
| ports_num_for_sparse  | 必选 | 1 | 和pserver之间用于稀疏类型参数通信的端口个数  |
| num_gradient_servers  | 必选 | 1 | 当前训练任务pserver总数 |
| trainer_id  | 必选 | 0 | 每个trainer的唯一ID，从0开始的整数 |
| pservers  | 必选 | 127.0.0.1 | 当前训练任务启动的pserver的IP列表，多个IP使用“,”隔开 |


### 准备数据集

参考样例数据准备脚本[prepare.py](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/howto/usage/cluster/src/word2vec/prepare.py)，准备训练数据和验证数据集，我们使用paddle.dataset.imikolov数据集，并根据分布式训练并发数（trainer节点个数），在`prepare.py`开头部分指定`SPLIT_COUNT`将数据切分成多份。

在线上系统中，通常会使用MapReduce任务的输出结果作为训练结果，这样训练文件的个数会比较多，而且个数并不确定。在trainer中可以使用下面取模的方法为每个trainer分配训练数据文件：

```python
import os
train_list = []
flist = os.listdir("/train_data/")
for f in flist:
  suffix = int(f.split("-")[1])
  if suffix % TRAINER_COUNT == TRAINER_ID:
    train_list.append(f)
```

示例程序`prepare.py`会把训练集和测试集分别分割成多个文件（例子中为3个，后缀为`-00000`、`-00001`和`-00002`）:
```
train.txt
train.txt-00000
train.txt-00001
train.txt-00002
test.txt
test.txt-00000
test.txt-00001
test.txt-00002
```

在进行分布式训练时，每个trainer进程需要能够读取属于自己的一份数据。在一些分布式系统中，系统会提供一个分布式存储服务，这样保存在分布式存储中的数据可以被集群中的每个节点读取到。如果不使用分布式存储，则需要手动拷贝属于每个trainer节点的训练数据到对应的节点上。

对于不同的训练任务，训练数据格式和训练程序的`reader()`会大不相同，所以开发者需要根据自己训练任务的实际场景完成训练数据的分割和`reader()`的编写。

### 准备训练程序

我们会对每个训练任务都会在每个节点上创建一个工作空间（workspace），其中包含了用户的训练程序、程序依赖、挂载或下载的训练数据分片。

最后，工作空间应如下所示：
```
.
|-- my_lib.py
|-- word_dict.pickle
|-- train.py
|-- train_data_dir/
|   |-- train.txt-00000
|   |-- train.txt-00001
|   |-- train.txt-00002
`-- test_data_dir/
    |-- test.txt-00000
    |-- test.txt-00001
    `-- test.txt-00002
```

- `my_lib.py`：会被`train.py`调用的一些用户定义的库函数，比如PIL库等。
- `word_dict.pickle`：在`train.py`中会使用到的字典数据文件。
- `train.py`：训练程序，代码参考[api_train_v2_cluster.py](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/howto/usage/cluster/src/word2vec/prepare.py)。***注意：*** 对于本样例代码，在使用不同的分布式计算平台时，您可能需要修改`train.py`开头的部分（如下），以便获得训练数据的位置和获取环境变量配置：

  ```python
  cluster_train_file = "./train_data_dir/train/train.txt"
  cluster_test_file = "./test_data_dir/test/test.txt"
  node_id = os.getenv("OMPI_COMM_WORLD_RANK")
  if not node_id:
      raise EnvironmentError("must provied OMPI_COMM_WORLD_RANK")
  ```

- `train_data_dir`：包含训练数据的目录，可以是从分布式存储挂载过来的，也可以是在任务启动前下载到本地的。
- `test_data_dir`：包含测试数据集的目录。

## 使用分布式计算平台或工具

PaddlePaddle可以使用多种分布式计算平台构建分布式计算任务，包括：
- [Kubernetes](http://kubernetes.io) Google开源的容器集群的调度框架，支持大规模集群生产环境的完整集群方案。
- [OpenMPI](https://www.open-mpi.org) 成熟的高性能并行计算框架。
- [Fabric](http://www.fabfile.org) 集群管理工具。可以使用`Fabric`编写集群任务提交和管理脚本。

对于不同的集群平台，会分别介绍集群作业的启动和停止方法。这些例子都可以在[cluster_train_v2](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/cluster_train_v2)找到。

在使用分布式计算平台进行训练时，任务被调度在集群中时，分布式计算平台通常会通过API或者环境变量提供任务运行需要的参数，比如节点的ID、IP和任务节点个数等。

### 使用Fabric启动集群作业

#### 准备一个Linux集群
可以在`paddle/scripts/cluster_train_v2/fabric/docker_cluster`目录下，执行`kubectl -f ssh_servers.yaml`启动一个测试集群，并使用`kubectl get po -o wide`获得这些节点的IP地址。

#### 启动集群作业

`paddle.py` 提供了自动化脚本来启动不同节点中的所有 PaddlePaddle 集群进程。默认情况下，所有命令行选项可以设置为 `paddle.py` 命令选项并且 `paddle.py` 将透明、自动地将这些选项应用到 PaddlePaddle 底层进程。

`paddle.py` 为方便作业启动提供了两个独特的命令选项。

-  `job_dispatch_package`  设为本地 `workspace` 目录，它将被分发到 `conf.py` 中设置的所有节点。它有助于帮助频繁修改和访问工作区文件的用户减少负担，否则频繁的多节点工作空间部署可能会很麻烦。
-  `job_workspace`  设为已部署的工作空间目录，`paddle.py` 将跳过分发阶段直接启动所有节点的集群作业。它可以帮助减少分发延迟。

`cluster_train/run.sh` 提供了命令样例来运行 `doc/howto/usage/cluster/src/word2vec` 集群任务，只需用您定义的目录修改 `job_dispatch_package` 和 `job_workspace`，然后：
```
sh run.sh
```

集群作业将会在几秒后启动。

#### 终止集群作业
`paddle.py`能获取`Ctrl + C` SIGINT 信号来自动终止它启动的所有进程。只需中断 `paddle.py` 任务来终止集群作业。如果程序崩溃你也可以手动终止。

#### 检查集群训练结果
详细信息请检查 $workspace/log 里的日志，每一个节点都有相同的日志结构。

`paddle_trainer.INFO`
提供几乎所有训练的内部输出日志，与本地训练相同。这里检验运行时间模型的收敛。

`paddle_pserver2.INFO`
提供 pserver 运行日志，有助于诊断分布式错误。

`server.log`
提供 parameter server 进程的 stderr 和 stdout。训练失败时可以检查错误日志。

`train.log`
提供训练过程的 stderr 和 stdout。训练失败时可以检查错误日志。

#### 检查模型输出
运行完成后，模型文件将被写入节点 0 的 `output` 目录中。
工作空间中的 `nodefile` 表示当前集群作业的节点 ID。

### 在OpenMPI集群中提交训练作业

#### 准备OpenMPI集群

执行下面的命令以启动3个节点的OpenMPI集群和一个"head"节点：

```bash
paddle/scripts/cluster_train_v2/openmpi/docker_cluster
kubectl create -f head.yaml
kubectl create -f mpi-nodes.yaml
```

然后可以从head节点ssh无密码登录到OpenMPI的每个节点上。

#### 启动集群作业

您可以按照下面的步骤在OpenMPI集群中提交paddle训练任务：

```bash
# 获得head和node节点的IP地址
kubectl get po -o wide
# 将node节点的IP地址保存到machines文件中
kubectl get po -o wide | grep nodes | awk '{print $6}' > machines
# 拷贝必要的文件到head节点
scp -i ssh/id_rsa.mpi.pub machines prepare.py train.py start_mpi_train.sh tutorial@[headIP]:~
# ssh 登录到head节点
ssh -i ssh/id_rsa.mpi.pub tutorial@[headIP]
# --------------- 以下操作均在head节点中执行 ---------------
# 准备训练数据
python prepare.py
# 拷贝训练程序和字典文件到每台MPI节点
cat machines | xargs -i scp word_dict.pickle train.py start_mpi_train.sh machines {}:/home/tutorial
# 创建日志目录
mpirun -hostfile machines -n 3 mkdir /home/tutorial/logs
# 拷贝训练数据到各自的节点
scp train.txt-00000 test.txt-00000 [node1IP]:/home/tutorial
scp train.txt-00001 test.txt-00001 [node2IP]:/home/tutorial
scp train.txt-00002 test.txt-00002 [node3IP]:/home/tutorial
# 启动训练任务
mpirun -hostfile machines -n 3  /home/tutorial/start_mpi_train.sh
```

### 在Kubernetes集群中提交训练作业

此部分的使用方法可以参考[here](../k8s/k8s_distributed_cn.md)。
