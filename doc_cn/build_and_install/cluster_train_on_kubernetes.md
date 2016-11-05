
# 使用Kubernetes进行分布式训练

>前一篇文章介绍了如何使用Kubernetes Job进行一次单机的Paddle训练。在这篇文档里，我们介绍如何使用 Kubernetes 进行Paddle的集群训练作业。
>关于Paddle的分布式集群训练，可以参考 [Cluster Training](https://github.com/baidu/Paddle/blob/develop/doc/cluster/opensource/cluster_train.md), 本文在此基础上，利用了Kubernetes快速构建Paddle集群，进行分布式训练任务。

## 制作镜像

Paddle的集群训练需要有一个Paddle集群来实现，在本文中，我们使用Kubernetes来快速创建一个Paddle集群。我们使用 `paddledev/paddle:cpu-demo-latest` 镜像作为Paddle集群节点的运行环境，里面包含了 Paddle 运行所需要的相关依赖，同时，为了能将训练任务及配置统一分发到各个节点，需要使用到`sshd`以便使用`fabric`来操作。镜像的 Dockerfile 如下：

```
FROM paddledev/paddle:cpu-demo-latest

RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd

RUN sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

EXPOSE 22

CMD    ["/usr/sbin/sshd", "-D"]
```

使用 `docker build` 构建镜像：

```
docker build -t mypaddle:paddle_demo_ssh .
```

## 准备工作空间

工作空间 [Job Workspace](https://github.com/baidu/Paddle/blob/develop/doc/cluster/opensource/cluster_train.md#prepare-job-workspace) , 即一个包含了依赖库，训练，测试数据，模型配置文件的目录。参考 [Cluster Training](https://github.com/baidu/Paddle/blob/develop/doc/cluster/opensource/cluster_train.md)中的例子，我们也是用`demo/recommendation`作为本文的训练任务。此demo可直接从[Github Paddle源码](https://github.com/baidu/Paddle/tree/develop/demo/recommendation)中获取。

### 准备训练数据

在Paddle源码中，找到`demo/recommendation`文件夹，即为我们的Workspace, 在本文的环境中，路径为`/home/work/paddle-demo/Paddle/demo/recommendation`

```
[root@paddle-k8s-node0 recommendation]# tree
.
├── common_utils.py
├── data
│   ├── config_generator.py
│   ├── config.json
│   ├── meta_config.json
│   ├── meta_generator.py
│   ├── ml_data.sh
│   └── split.py
├── dataprovider.py
├── evaluate.sh
├── prediction.py
├── preprocess.sh
├── requirements.txt
├── run.sh
└── trainer_config.py

1 directory, 14 files
```

运行`data/ml_data.sh`脚本，下载数据，然后运行`preprocess.sh`脚本进行预处理。

```
[root@paddle-k8s-node0 recommendation]# data/ml_data.sh
++ dirname data/ml_data.sh
+ cd data
+ wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
--2016-11-04 10:14:49--  http://files.grouplens.org/datasets/movielens/ml-1m.zip
Resolving files.grouplens.org (files.grouplens.org)... 128.101.34.146
Connecting to files.grouplens.org (files.grouplens.org)|128.101.34.146|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 5917549 (5.6M) [application/zip]
Saving to: ‘ml-1m.zip’

100%[==========================>] 5,917,549   50.6KB/s   in 2m 29s

2016-11-04 10:17:20 (38.8 KB/s) - ‘ml-1m.zip’ saved [5917549/5917549]

+ unzip ml-1m.zip
Archive:  ml-1m.zip
   creating: ml-1m/
  inflating: ml-1m/movies.dat
  inflating: ml-1m/ratings.dat
  inflating: ml-1m/README
  inflating: ml-1m/users.dat
+ rm ml-1m.zip

[root@paddle-k8s-node0 recommendation]# ./preprocess.sh
generate meta config file
generate meta file
split train/test file
shuffle train file
```

### 修改集群训练配置

参考[Cluster Training](https://github.com/baidu/Paddle/blob/develop/doc/cluster/opensource/cluster_train.md)中的介绍，我们使用`paddle/scripts/cluster_train/`中的文件来作为分布式训练任务的配置和启动脚本。在`run.sh`文件中，填入我们的workspace和训练配置文件路径。

```
#!/bin/sh
python paddle.py \
  --job_dispatch_package="/home/work/paddle-demo/Paddle/demo/recommendation" \
  --dot_period=10 \
  --ports_num_for_sparse=2 \
  --log_period=50 \
  --num_passes=10 \
  --trainer_count=4 \
  --saving_period=1 \
  --local=0 \
  --config=/home/work/paddle-demo/Paddle/demo/recommendation/trainer_config.py \
  --save_dir=./output \
  --use_gpu=0
```

## 创建Paddle集群

创建Paddle集训需要编写创建Kubernetes资源的yaml文件，首先，创建一个Service,便于我们通过此Service来查找其对应的Paddle节点。

```
apiVersion: v1
kind: Service
metadata:
  name: cluster-demo
spec:
  selector:
    app: cluster-demo
  ports:
  - name: default
    protocol: TCP
    port: 7164
    targetPort: 7164
```

为了创建多个Paddle节点,我们使用Kubernetes ReplicationController资源来控制Paddle集群中的节点数量，Paddle节点之间需要开放相关的端口来互相通信。下面的例子中，我们开放了每个Paddle节点的7164-7167端口，例如，一个包含4个节点的Paddle集群的yaml文件如下：

```
apiVersion: v1
kind: ReplicationController
metadata:
  name: cluster-demo
spec:
  replicas: 4
  selector:
    app: cluster-demo
  template:
    metadata:
      name: cluster-demo
      labels:
        app: cluster-demo
    spec:
      containers:
      - name: cluster-demo
        image: mypaddle:paddle_demo_ssh
        ports:
        - containerPort: 7164
        - containerPort: 7165
        - containerPort: 7166
        - containerPort: 7167
```

然后我们可以通过`kubectl`工具来查看所创建的资源信息。

首先查看我们创建的Paddle Service，然后根据Service,查看所创建的Paddle节点的IP地址。

```
[root@paddle-k8s-node0 cluster_train]# kubectl  get svc
NAME                   CLUSTER-IP   EXTERNAL-IP   PORT(S)    AGE
cluster-demo           11.1.1.77    <none>        7164/TCP   6h

[root@paddle-k8s-node0 cluster_train]# kubectl get -o json endpoints cluster-demo | grep ip
                    "ip": "192.168.129.79",
                    "ip": "192.168.129.80",
                    "ip": "192.168.223.157",
                    "ip": "192.168.223.158",
```

## 开始集群训练

我们需要在`paddle/scripts/cluster_train/conf.py`文件中指定各个节点的IP地址以及开放的端口。根据上文创建的信息，`conf.py`文件修改如下：

```
HOSTS = [
        "root@192.168.129.79",
        "root@192.168.129.80",
        "root@192.168.223.157",
        "root@192.168.223.158"
        ]

'''
workspace configuration
'''
#root dir for workspace, can be set as any director with real user account
ROOT_DIR = "/home/paddle"


'''
network configuration
'''
#pserver nics
PADDLE_NIC = "eth0"
#pserver port
PADDLE_PORT = 7164
#pserver ports num
PADDLE_PORTS_NUM = 2
#pserver sparse ports num
PADDLE_PORTS_NUM_FOR_SPARSE = 2

#environments setting for all processes in cluster job
LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib64"
```
然后使用`run.sh`脚本开始训练，启动的打印如下：

```
[root@paddle-k8s-node0 cluster_train]# ./run.sh
[root@192.168.129.79] Executing task 'job_create_workspace'
......
[root@192.168.129.80] Executing task 'job_create_workspace'
......
[root@192.168.223.157] Executing task 'job_create_workspace'
......
[root@192.168.223.158] Executing task 'job_create_workspace'
......
[root@192.168.129.79] run: echo 0 > /home/paddle/JOB20161104171630/nodefile
[root@192.168.129.80] Executing task 'set_nodefile'
[root@192.168.129.80] run: echo 1 > /home/paddle/JOB20161104171630/nodefile
[root@192.168.223.157] Executing task 'set_nodefile'
[root@192.168.223.157] run: echo 2 > /home/paddle/JOB20161104171630/nodefile
[root@192.168.223.158] Executing task 'set_nodefile'
[root@192.168.223.158] run: echo 3 > /home/paddle/JOB20161104171630/nodefile
```

可以看到192.168.129.79，192.168.129.80，192.168.223.157，192.168.223.158分别为Paddle集群的Node 0-3.

我们可以进入其中一个Paddle节点查看训练的日志。

```
root@cluster-demo-fwwi5:/home/paddle/JOB20161104171700/log# less paddle_trainer.INFO
Log file created at: 2016/11/04 09:17:20
Running on machine: cluster-demo-fwwi5
Log line format: [IWEF]mmdd hh:mm:ss.uuuuuu threadid file:line] msg
I1104 09:17:20.346797   108 Util.cpp:155] commandline: /usr/local/bin/../opt/paddle/bin/paddle
_trainer --num_gradient_servers=4 --nics=eth0 --port=7164 --ports_num=2 --comment=paddle_proce
ss_by_paddle --pservers=192.168.129.79,192.168.129.80,192.168.223.157,192.168.223.158 --ports_
num_for_sparse=2 --config=./trainer_config.py --trainer_count=4 --use_gpu=0 --num_passes=10 --
save_dir=./output --log_period=50 --dot_period=10 --saving_period=1 --local=0 --trainer_id=1

root@cluster-demo-fwwi5:/home/paddle/JOB20161104171700/log# tailf paddle_trainer.INFO
......
I1104 09:17:37.376471   150 ThreadLocal.cpp:37] thread use undeterministic rand seed:151
I1104 09:18:54.159624   108 TrainerInternal.cpp:163]  Batch=50 samples=80000 AvgCost=4.03478 CurrentCost=4.03478 Eval:  CurrentEval:

I1104 09:20:10.207902   108 TrainerInternal.cpp:163]  Batch=100 samples=160000 AvgCost=3.75806 CurrentCost=3.48134 Eval:  CurrentEval:
I1104 09:21:26.493571   108 TrainerInternal.cpp:163]  Batch=150 samples=240000 AvgCost=3.64512 CurrentCost=3.41923 Eval:  CurrentEval:

```

最后，我们可以在Paddle集群的node0（192.168.129.79）上查看训练的输出结果。

```
[root@paddle-k8s-node0 ~]# ssh root@192.168.129.79
......
root@cluster-demo-r65g0:/home/paddle/JOB20161104171700/output/pass-00000# ll
total 14876
drwxr-xr-x. 2 root root    4096 Nov  4 09:40 ./
drwxr-xr-x. 3 root root      23 Nov  4 09:40 ../
-rw-r--r--. 1 root root 4046864 Nov  4 09:40 ___embedding_0__.w0
-rw-r--r--. 1 root root  100368 Nov  4 09:40 ___embedding_1__.w0
-rw-r--r--. 1 root root 6184976 Nov  4 09:40 ___embedding_2__.w0
-rw-r--r--. 1 root root    2064 Nov  4 09:40 ___embedding_3__.w0
-rw-r--r--. 1 root root    7184 Nov  4 09:40 ___embedding_4__.w0
-rw-r--r--. 1 root root   21520 Nov  4 09:40 ___embedding_5__.w0
-rw-r--r--. 1 root root  262160 Nov  4 09:40 ___fc_layer_0__.w0
-rw-r--r--. 1 root root    1040 Nov  4 09:40 ___fc_layer_0__.wbias
......
......
-rw-r--r--. 1 root root  262160 Nov  4 09:40 _movie_fusion.w0
-rw-r--r--. 1 root root  262160 Nov  4 09:40 _movie_fusion.w1
-rw-r--r--. 1 root root  262160 Nov  4 09:40 _movie_fusion.w2
-rw-r--r--. 1 root root    1040 Nov  4 09:40 _movie_fusion.wbias
-rw-r--r--. 1 root root  262160 Nov  4 09:40 _user_fusion.w0
-rw-r--r--. 1 root root  262160 Nov  4 09:40 _user_fusion.w1
-rw-r--r--. 1 root root  262160 Nov  4 09:40 _user_fusion.w2
-rw-r--r--. 1 root root  262160 Nov  4 09:40 _user_fusion.w3
-rw-r--r--. 1 root root    1040 Nov  4 09:40 _user_fusion.wbias
-rw-r--r--. 1 root root     169 Nov  4 09:40 done
-rw-r--r--. 1 root root      17 Nov  4 09:40 path.txt
-rw-r--r--. 1 root root    3495 Nov  4 09:40 trainer_config.py
```