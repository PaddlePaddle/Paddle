# Kubernetes单机训练

在这篇文档里，我们介绍如何在 Kubernetes 集群上启动一个单机使用CPU的PaddlePaddle训练作业。在下一篇中，我们将介绍如何启动分布式训练作业。

## 制作Docker镜像

在一个功能齐全的Kubernetes机群里，通常我们会安装Ceph等分布式文件系统来存储训练数据。这样的话，一个分布式PaddlePaddle训练任务中
的每个进程都可以从Ceph读取数据。在这个例子里，我们只演示一个单机作业，所以可以简化对环境的要求，把训练数据直接放在
PaddlePaddle的Docker Image里。为此，我们需要制作一个包含训练数据的PaddlePaddle镜像。

PaddlePaddle的 `paddlepaddle/paddle:cpu-demo-latest` 镜像里有PaddlePaddle的源码与demo，
（请注意，默认的PaddlePaddle生产环境镜像 `paddlepaddle/paddle:latest` 是不包括源码的，PaddlePaddle的各版本镜像可以参考
[Docker Installation Guide](http://paddlepaddle.org/docs/develop/documentation/zh/getstarted/build_and_install/docker_install_cn.html)），
下面我们使用这个镜像来下载数据到Docker Container中，并把这个包含了训练数据的Container保存为一个新的镜像。

### 运行容器

```
$ docker run --name quick_start_data -it paddlepaddle/paddle:cpu-demo-latest
```

### 下载数据

进入容器`/root/paddle/demo/quick_start/data`目录，使用`get_data.sh`下载数据

```
$ root@fbd1f2bb71f4:~/paddle/demo/quick_start/data# ./get_data.sh

Downloading Amazon Electronics reviews data...
--2016-10-31 01:33:43--  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
Resolving snap.stanford.edu (snap.stanford.edu)... 171.64.75.80
Connecting to snap.stanford.edu (snap.stanford.edu)|171.64.75.80|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 495854086 (473M) [application/x-gzip]
Saving to: 'reviews_Electronics_5.json.gz'

 10% [=======>                                         ] 874,279     64.7KB/s  eta 2h 13m

```

### 修改启动脚本

下载完数据后，修改`/root/paddle/demo/quick_start/train.sh`文件，内容如下（增加了一条cd命令）
```
set -e
cd /root/paddle/demo/quick_start
cfg=trainer_config.lr.py
#cfg=trainer_config.emb.py
#cfg=trainer_config.cnn.py
#cfg=trainer_config.lstm.py
#cfg=trainer_config.bidi-lstm.py
#cfg=trainer_config.db-lstm.py
paddle train \
  --config=$cfg \
  --save_dir=./output \
  --trainer_count=4 \
  --log_period=20 \
  --num_passes=15 \
  --use_gpu=false \
  --show_parameter_stats_period=100 \
  --test_all_data_in_one_period=1 \
  2>&1 | tee 'train.log'
```

### 提交镜像

修改启动脚本后，退出容器，使用`docker commit`命令创建新镜像。

```
$ docker commit quick_start_data mypaddle/paddle:quickstart
```

## 使用 Kubernetes 进行训练

>针对任务运行完成后容器自动退出的场景，Kubernetes有Job类型的资源来支持。下文就是用Job类型的资源来进行训练。

### 编写yaml文件

在训练时，输出结果可能会随着容器的消耗而被删除，需要在创建容器前挂载卷以便我们保存训练结果。使用我们之前构造的镜像，可以创建一个 [Kubernetes Job](http://kubernetes.io/docs/user-guide/jobs/#what-is-a-job)，简单的yaml文件如下：

```
apiVersion: batch/v1
kind: Job
metadata:
  name: quickstart
spec:
  parallelism: 1
  completions: 1
  template:
    metadata:
      name: quickstart
    spec:
      volumes:
      - name: output
        hostPath: 
          path: /home/work/paddle_output     
      containers:
      - name: pi
        image: mypaddle/paddle:quickstart
        command: ["bin/bash",  "-c", "/root/paddle/demo/quick_start/train.sh"]
        volumeMounts:
        - name: output
          mountPath: /root/paddle/demo/quick_start/output
      restartPolicy: Never
```

### 创建PaddlePaddle Job

使用上文创建的yaml文件创建Kubernetes Job，命令为：

```
$ kubectl  create -f paddle.yaml
```

查看job的详细情况：

```
$ kubectl  get job
NAME         DESIRED   SUCCESSFUL   AGE
quickstart   1         0            58s

$ kubectl  describe job quickstart
Name:		quickstart
Namespace:	default
Image(s):	registry.baidu.com/public/paddle:cpu-demo-latest
Selector:	controller-uid=f120da72-9f18-11e6-b363-448a5b355b84
Parallelism:	1
Completions:	1
Start Time:	Mon, 31 Oct 2016 11:20:16 +0800
Labels:		controller-uid=f120da72-9f18-11e6-b363-448a5b355b84,job-name=quickstart
Pods Statuses:	0 Running / 1 Succeeded / 0 Failed
Volumes:
  output:
    Type:	HostPath (bare host directory volume)
    Path:	/home/work/paddle_output
Events:
  FirstSeen	LastSeen	Count	From			SubobjectPath	Type		Reason			Message
  ---------	--------	-----	----			-------------	--------	------			-------
  1m		1m		1	{job-controller }			Normal		SuccessfulCreate	Created pod: quickstart-fa0wx
```

### 查看训练结果

根据Job对应的Pod信息，可以查看此Pod运行的宿主机。

```
kubectl  describe pod quickstart-fa0wx
Name:		quickstart-fa0wx
Namespace:	default
Node:		paddle-demo-let02/10.206.202.44
Start Time:	Mon, 31 Oct 2016 11:20:17 +0800
Labels:		controller-uid=f120da72-9f18-11e6-b363-448a5b355b84,job-name=quickstart
Status:		Succeeded
IP:		10.0.0.9
Controllers:	Job/quickstart
Containers:
  quickstart:
    Container ID:	docker://b8561f5c79193550d64fa47418a9e67ebdd71546186e840f88de5026b8097465
    Image:		registry.baidu.com/public/paddle:cpu-demo-latest
    Image ID:		docker://18e457ce3d362ff5f3febf8e7f85ffec852f70f3b629add10aed84f930a68750
    Port:
    Command:
      bin/bash
      -c
      /root/paddle/demo/quick_start/train.sh
    QoS Tier:
      cpu:		BestEffort
      memory:		BestEffort
    State:		Terminated
      Reason:		Completed
      Exit Code:	0
      Started:		Mon, 31 Oct 2016 11:20:20 +0800
      Finished:		Mon, 31 Oct 2016 11:21:46 +0800
    Ready:		False
    Restart Count:	0
    Environment Variables:
Conditions:
  Type		Status
  Ready 	False
Volumes:
  output:
    Type:	HostPath (bare host directory volume)
    Path:	/home/work/paddle_output
```

我们还可以登录到宿主机上查看训练结果。

```
[root@paddle-demo-let02 paddle_output]# ll
total 60
drwxr-xr-x 2 root root 4096 Oct 31 11:20 pass-00000
drwxr-xr-x 2 root root 4096 Oct 31 11:20 pass-00001
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00002
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00003
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00004
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00005
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00006
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00007
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00008
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00009
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00010
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00011
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00012
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00013
drwxr-xr-x 2 root root 4096 Oct 31 11:21 pass-00014
```
