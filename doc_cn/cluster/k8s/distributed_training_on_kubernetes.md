
# Paddle on Kubernetes：分布式训练

前一篇文章介绍了如何在Kubernetes集群上启动一个单机Paddle训练作业 (Job)。在这篇文章里，我们介绍如何在Kubernetes集群上进行分布式Paddle训练作业。关于Paddle的分布式训练，可以参考 [Cluster Training](https://github.com/baidu/Paddle/blob/develop/doc/cluster/opensource/cluster_train.md), 本文利用Kubernetes的调度功能与容器编排能力，快速构建Paddle容器集群，进行分布式训练任务。

## Kubernetes 基本概念

在介绍分布式训练之前，需要对Kubernetes(k8s)有一个基本的认识，下面先简要介绍一下本文用到的几个k8s概念。

### Node

[`Node`](http://kubernetes.io/docs/admin/node/) 表示一个k8s集群中的一个工作节点，这个节点可以是物理机或者虚拟机，k8s集群就是由`node`节点与`master`节点组成的。每个node都安装有Docker,在本文的例子中，`Paadle`容器就在node上运行。

### Pod

一个[`Pod`](http://kubernetes.io/docs/user-guide/pods/) 是一组(一个或多个)容器，pod是k8s的最小调度单元，一个pod中的所有容器会被调度到同一个node上。Pod中的容器共享NET,PID,IPC,UTS等Linux namespace,它们使用同一个IP地址，可以通过`localhost`互相通信。不同pod之间可以通过IP地址访问。

### Job

[`Job`](http://kubernetes.io/docs/user-guide/jobs/) 可以翻译为作业，每个job可以设定pod成功完成的次数,一次作业会创建多个pod，当成功完成的pod个数达到预设值时，就表示job成功结束了。

### Volume

[`Volume`](http://kubernetes.io/docs/user-guide/volumes/) 存储卷，是pod内的容器都可以访问的共享目录，也是容器与node之间共享文件的方式，因为容器内的文件都是暂时存在的，当容器因为各种原因被销毁时，其内部的文件也会随之消失。通过volume,就可以将这些文件持久化存储。k8s支持多种volume,例如`hostPath(宿主机目录)`，`gcePersistentDisk`，`awsElasticBlockStore`等。

### Namespace

[`Namespaces`](http://kubernetes.io/docs/user-guide/volumes/) 命名空间，在k8s中创建的所有资源对象(例如上文的pod,job)等都属于一个命名空间，在同一个命名空间中，资源对象的名字是唯一的，不同空间的资源名可以重复，命名空间主要用来为不同的用户提供相对隔离的环境。本文只使用了`default`默认命名空间，读者可以不关心此概念。

## 整体方案

### 前提条件

首先，我们需要拥有一个k8s集群，在这个集群中所有node与pod都可以互相通信。关于k8s集群搭建，可以参考[官方文档](http://kubernetes.io/docs/getting-started-guides/kubeadm/)，在以后的文章中我们也会介绍AWS上搭建的方案。在本文的环境中，k8s集群中所有node都挂载了一个`mfs`(分布式文件系统)共享目录，我们通过这个目录来存放训练文件与最终输出的模型。在训练之前，用户将配置与训练数据切分好放在mfs目录中，训练时，程序从此目录拷贝文件到容器内进行训练，将结果保存到此目录里。

### 使用 `Job`

我们使用k8s中的job这个概念来代表一次分布式训练。`Job`表示一次性作业，在作业完成后，k8s会销毁job产生的容器并且释放相关资源。

在k8s中，可以通过编写一个 `yaml` 文件，来描述这个job，在这个文件中，主要包含了一些配置信息，例如Paddle节点的个数，`paddle pserver`开放的端口个数与端口号，`paddle`使用的网卡设备等，这些信息通过环境变量的形式传递给容器内的程序使用。

在一次分布式训练中，用户确定好本次训练需要的Paddle节点个数，将切分好的训练数据与配置文件上传到`mfs`共享目录中。然后编写这次训练的`job yaml`文件,提交给k8s集群创建并开始作业。

### 创建`Paddle`节点

当k8s master收到`job yaml`文件后，会解析相关字段，创建出多个pod(个数为Paddle节点数)，k8s会把这些pod调度到集群的node上运行。一个`pod`就代表一个`Paddle`节点，当pod被成功分配到一台物理/虚拟机上后，k8s会启动pod内的容器，这个容器会根据`job yaml`文件中的环境变量，启动`paddle pserver`与`paddle train`进程。

### 启动训练

在容器启动后，会通过脚本来启动这次分布式训练，我们知道`paddle train`进程启动时需要知道其他节点的IP地址以及本节点的`trainer_id`，由于`Paddle`本身不提供类似服务发现的功能，所以在本文的启动脚本中，每个节点会根据`job name`向`k8s apiserver`查询这个`job`对应的所有`pod`信息(k8s默认会在每个容器的环境变量中写入`apiserver`的地址)。

根据这些pod信息，就可以通过某种方式，为每个pod分配一个唯一的`trainer_id`。本文把所有pod的IP地址进行排序，将顺序作为每个`Paddle`节点的`trainer_id`。启动脚本的工作流程大致如下：

  1. 查询`k8s apiserver`获取pod信息，根据IP分配`trainer_id`
  1. 从`mfs`共享目录中拷贝训练文件到容器内
  1. 根据环境变量，解析出`paddle pserver`与`paddle train`的启动参数，启动进程
  1. 训练时，`Paddle`会自动将结果保存在`trainer_id`为0的节点上，将输出路径设置为`mfs`目录，保存输出的文件


## 搭建过程

根据前文的描述，要在已有的k8s集群上进行`Paddle`的分布式训练，主要分为以下几个步骤：

1. 制作`Paddle`镜像
1. 将训练文件与切分好的数据上传到共享存储
1. 编写本次训练的`job yaml`文件，创建`k8s job`
1. 训练结束后查看输出结果

下面就根据这几个步骤分别介绍。



### 制作镜像

`Paddle`镜像需要提供`paddle pserver`与`paddle train`进程的运行环境，用这个镜像创建的容器需要有以下两个功能：

- 拷贝训练文件到容器内

- 生成`paddle pserver`与`paddle train`进程的启动参数，并且启动训练

因为官方镜像 `paddledev/paddle:cpu-latest` 内已经包含`Paddle`的执行程序但是还没上述功能，所以我们可以在这个基础上，添加启动脚本，制作新镜像来完成以上的工作。镜像的`Dockerfile`如下：

```Dockerfile
FROM paddledev/paddle:cpu-latest

MAINTAINER zjsxzong89@gmail.com

COPY start.sh /root/
COPY start_paddle.py /root/
CMD ["bash"," -c","/root/start.sh"]
```

[`start.sh`](start.sh)文件拷贝训练文件到容器内，然后执行[`start_paddle.py`](start_paddle.py)脚本启动训练，前文提到的获取其他节点IP地址，分配`trainer_id`等都在`start_paddle.py`脚本中完成。


使用 `docker build` 构建镜像：

```bash
docker build -t registry.baidu.com/public/paddle:mypaddle .
```

然后将构建成功的镜像上传到镜像仓库，注意本文中使用的`registry.baidu.com`是一个私有仓库，读者可以根据自己的情况部署私有仓库或者使用`Docker hub`。

```bash
docker push  registry.baidu.com/public/paddle:mypaddle
```

### 上传训练文件

本文使用`Paddle`官方的`recommendation demo`作为这次训练的内容，我们将训练文件与数据放在一个`job name`命名的目录中，上传到`mfs`共享存储。完成后`mfs`上的文件内容大致如下：

```bash
[root@paddle-k8s-node0 mfs]# tree -d
.
└── paddle-cluster-job
    ├── data
    │   ├── 0
    │   │
    │   ├── 1
    │   │
    │   └── 2
    ├── output
    └── recommendation
```

目录中`paddle-cluster-job`是本次训练对应的`job name`，本次训练要求有3个`Paddle`节点，在`paddle-cluster-job/data`目录中存放切分好的数据，文件夹`0,1,2`分别代表3个节点的`trainer_id`。`recommendation`文件夹内存放训练文件，`output`文件夹存放训练结果与日志。

### 创建`job`

`k8s`可以通过`yaml`文件来创建相关对象，然后可以使用命令行工具创建`job`。

`job yaml`文件描述了这次训练使用的Docker镜像,需要启动的节点个数以及 `paddle pserver`与 `paddle train`进程启动的必要参数，也描述了容器需要使用的存储卷挂载的情况。`yaml`文件中各个字段的具体含义，可以查看[`k8s官方文档`](http://kubernetes.io/docs/api-reference/batch/v1/definitions/#_v1_job)。例如，本次训练的`yaml`文件可以写成：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: paddle-cluster-job
spec:
  parallelism: 3
  completions: 3
  template:
    metadata:
      name: paddle-cluster-job
    spec:
      volumes:
      - name: jobpath
        hostPath:
          path: /home/work/mfs
      containers:
      - name: trainer
        image: registry.baidu.com/public/paddle:mypaddle
        command: ["bin/bash",  "-c", "/root/start.sh"]
        env:
        - name: JOB_NAME
          value: paddle-cluster-job
        - name: JOB_PATH
          value: /home/jobpath
        - name: JOB_NAMESPACE
          value: default
        - name: TRAIN_CONFIG_DIR
          value: recommendation
        - name: CONF_PADDLE_NIC
          value: eth0
        - name: CONF_PADDLE_PORT
          value: "7164"
        - name: CONF_PADDLE_PORTS_NUM
          value: "2"
        - name: CONF_PADDLE_PORTS_NUM_SPARSE
          value: "2"
        - name: CONF_PADDLE_GRADIENT_NUM
          value: "3"
        volumeMounts:
        - name: jobpath
          mountPath: /home/jobpath
      restartPolicy: Never
```

文件中，`metadata`下的`name`表示这个`job`的名字。`parallelism,completions`字段表示这个`job`会同时开启3个`Paddle`节点，成功训练且退出的`pod`数目为3时，这个`job`才算成功结束。然后申明一个存储卷`jobpath`，代表宿主机目录`/home/work/mfs`，在对容器的描述`containers`字段中，将此目录挂载为容器的`/home/jobpath`目录，这样容器的`/home/jobpath`目录就成为了共享存储，放在这个目录里的文件其实是保存到了`mfs`上。

`env`字段表示容器的环境变量，我们将`paddle`运行的一些参数通过这种方式传递到容器内。

`JOB_PATH`表示共享存储挂载的路径，`JOB_NAME`表示job名字，`TRAIN_CONFIG_DIR`表示本次训练文件所在目录，这三个变量组合就可以找到本次训练需要的文件路径。

`CONF_PADDLE_NIC`表示`paddle pserver`进程需要的`--nics`参数，即网卡名

`CONF_PADDLE_PORT`表示`paddle pserver`的`--port`参数，`CONF_PADDLE_PORTS_NUM`则表示稠密更新的端口数量，也就是`--ports_num`参数。

`CONF_PADDLE_PORTS_NUM_SPARSE`表示稀疏更新的端口数量，也就是`--ports_num_for_sparse`参数。

`CONF_PADDLE_GRADIENT_NUM`表示训练节点数量，即`--num_gradient_servers`参数

编写完`yaml`文件后，可以使用k8s的命令行工具创建`job`.

```bash
kubectl create -f job.yaml
```

创建成功后，k8s就会创建3个`pod`作为`Paddle`节点然后拉取镜像，启动容器开始训练。


### 查看输出

在训练过程中，可以在共享存储上查看输出的日志和模型，例如`output`目录下就存放了输出结果。注意`node_0`，`node_1`，`node_2`这几个目录表示`Paddle`节点与`trainer_id`,并不是k8s中的`node`概念。

```bash
[root@paddle-k8s-node0 output]# tree -d
.
├── node_0
│   ├── server.log
│   └── train.log
├── node_1
│   ├── server.log
│   └── train.log
├── node_2
......
├── pass-00002
│   ├── done
│   ├── ___embedding_0__.w0
│   ├── ___embedding_1__.w0
......
```

我们可以通过日志查看容器训练的情况，例如：

```bash
[root@paddle-k8s-node0 node_0]# cat train.log
I1116 09:10:17.123121    50 Util.cpp:155] commandline:
 /usr/local/bin/../opt/paddle/bin/paddle_trainer
    --nics=eth0 --port=7164
    --ports_num=2 --comment=paddle_process_by_paddle
    --pservers=192.168.129.66,192.168.223.143,192.168.129.71
    --ports_num_for_sparse=2 --config=./trainer_config.py
    --trainer_count=4 --num_passes=10 --use_gpu=0 
    --log_period=50 --dot_period=10 --saving_period=1 
    --local=0 --trainer_id=0
    --save_dir=/home/jobpath/paddle-cluster-job/output
I1116 09:10:17.123440    50 Util.cpp:130] Calling runInitFunctions
I1116 09:10:17.123764    50 Util.cpp:143] Call runInitFunctions done.
[WARNING 2016-11-16 09:10:17,227 default_decorators.py:40] please use keyword arguments in paddle config.
[INFO 2016-11-16 09:10:17,239 networks.py:1282] The input order is [movie_id, title, genres, user_id, gender, age, occupation, rating]
[INFO 2016-11-16 09:10:17,239 networks.py:1289] The output order is [__regression_cost_0__]
I1116 09:10:17.392917    50 Trainer.cpp:170] trainer mode: Normal
I1116 09:10:17.613910    50 PyDataProvider2.cpp:257] loading dataprovider dataprovider::process
I1116 09:10:17.680917    50 PyDataProvider2.cpp:257] loading dataprovider dataprovider::process
I1116 09:10:17.681543    50 GradientMachine.cpp:134] Initing parameters..
I1116 09:10:18.012390    50 GradientMachine.cpp:141] Init parameters done.
I1116 09:10:18.018641    50 ParameterClient2.cpp:122] pserver 0 192.168.129.66:7164
I1116 09:10:18.018950    50 ParameterClient2.cpp:122] pserver 1 192.168.129.66:7165
I1116 09:10:18.019069    50 ParameterClient2.cpp:122] pserver 2 192.168.223.143:7164
I1116 09:10:18.019492    50 ParameterClient2.cpp:122] pserver 3 192.168.223.143:7165
I1116 09:10:18.019716    50 ParameterClient2.cpp:122] pserver 4 192.168.129.71:7164
I1116 09:10:18.019836    50 ParameterClient2.cpp:122] pserver 5 192.168.129.71:7165
```