# Kubernetes

In this article, we will introduce how to run PaddlePaddle training job on single CPU machine using Kubernetes. In next article, we will introduce how to run PaddlePaddle training job on distributed cluster.

## Build Docker Image

In distributed Kubernetes cluster, we will use Ceph or other distributed
storage system for storing training related data so that all processes in
PaddlePaddle training can retrieve data from Ceph. In this example, we will
only demo training job on single machine. In order to simplify the requirement
of the environment, we will directly put training data into the PaddlePaddle Docker Image,
so we need to create a PaddlePaddle Docker image that includes the training data.

The production Docker Image `paddlepaddle/paddle:cpu-demo-latest` has the PaddlePaddle
source code and demo. (Caution: Default PaddlePaddle Docker Image `paddlepaddle/paddle:latest` doesn't include
the source code, PaddlePaddle's different versions of Docker Image can be referred here:
[Docker Installation Guide](http://paddlepaddle.org/docs/develop/documentation/zh/getstarted/build_and_install/docker_install_en.html)),
so we run this Docker Image and download the training data, and then commit the whole
Container to be a new Docker Image.

### Run Docker Container

```
$ docker run --name quick_start_data -it paddlepaddle/paddle:cpu-demo-latest
```

### Download Training Data

Getting into `/root/paddle/demo/quick_start/data` Directory，using `get_data.sh` to download training data.
Then getting into `/root/paddle/demo/quick_start` Directory, using `preprocess.sh` to pre-process training data.

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

### Modify Startup Script

After downloading the data，modify `/root/paddle/demo/quick_start/train.sh` file contents are as follows (one more cd cmd):
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

### Commit Docker Image

```
$ docker commit quick_start_data mypaddle/paddle:quickstart
```

## Use Kubernetes For Training

We will use Kubernetes job for training process, following steps shows how to do the training with Kubernetes.

### Create Yaml Files

The output result in container will be demolished when job finished (container stopped running), so we need to mount the volume out to the local disk when creating the container to store the training result. Using our previously created image, we can create a [Kubernetes Job](http://kubernetes.io/docs/user-guide/jobs/#what-is-a-job), the yaml contents are as follows:

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

### Start PaddlePaddle Job

Using the above yaml file to start the Kubernetes job.

```
$ kubectl  create -f paddle.yaml
```

Get the detailed status of the job:

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

### Get Training Result

We can use kubectl command to take a look at the status of related pod.

```
$ kubectl  describe pod quickstart-fa0wx
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

We can also ssh to Kubernetes node to take a look at the training result.

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
