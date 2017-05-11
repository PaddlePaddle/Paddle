* [Introduction](#introduction)
* [Preparations](#preparations)
* [Command-line arguments](#command-line-arguments)
   * [Starting parameter server](#starting-parameter-server)
   * [Starting trainer](#starting-trainer)
   * [Prepare Training Dataset](#prepare-training-dataset)
   * [Prepare Training program](#prepare-training-program)
* [Use cluster platforms or cluster management tools](#use-cluster-platforms-or-cluster-management-tools)
   * [Cluster Training Using Fabric](#cluster-training-using-fabric)
      * [Prepare a demo cluster using kubernetes(optional)](#prepare-a-demo-cluster-using-kubernetesoptional)
      * [Launching Cluster Job](#launching-cluster-job)
      * [Kill Cluster Job](#kill-cluster-job)
      * [Check Cluster Training Result](#check-cluster-training-result)
      * [Check Model Output](#check-model-output)
   * [Cluster Training Using OpenMPI](#cluster-training-using-openmpi)
      * [Prepare a demo OpenMPI cluster using kubernetes(optional)](#prepare-a-demo-openmpi-cluster-using-kubernetesoptional)
      * [Launching Cluster Job](#launching-cluster-job-1)
   * [Cluster Training Using Kubernetes](#cluster-training-using-kubernetes)
# Introduction

In this article, we'll explain how to do distributed training jobs with PaddlePaddle on different types of clusters. The diagram below shows the mail architecture of a distributed trainning job:

<img src="src/trainer.png" width="500"/>

- Data shard: training data will be split into multiple parts, trainers use some parts of the whole dataset to do the training job.
- Trainer: each trainer reads the data shard, and do Neural Network training algorithms like "forward" and "backward". Then the trainer will upload calculated "gradients" to parameter servers, and wait for parameters to be optimized on the parameter server side. When that finishes, the trainer download optimized parameters and continues its training.
- Parameter server: every parameter server stores part of the whole Neural Network model data. They will do optimization calculations when gradients are uploaded from trainers, and then send updated parameters to trainers.

Trainers and Parameter servers coordinate form the distributed SGD(stochastic gradient descent) training algorithm. Meanwhile, PaddlePaddle can support both synchronize SGD and asynchronous SGD.

When training with synchronize SGD, PaddlePaddle uses an internal "synchronize barrier" which makes gradients update and parameter download in strict order. On the other hand, asynchronous SGD won't wait for all trainers to finish upload at a single step, this will increase the parallelism of distributed training: parameter servers do not depend on each other, they'll do parameter optimization concurrently. Parameter servers will not wait for trainers, so trainers will also do their work concurrently. But asynchronous SGD will introduce more randomness and noises.

# Preparations
1. Ready your computer cluster. They are normally a bunch of Linux servers connected by LAN. Each server will be assigned a unique IP address. The computer in the cluster can be called "nodes". Different hardware setup may cause some issue when using different cluster management platforms.
2. Install PaddlePaddle on every node. If you are going to take advantage of GPU cards, you'll also need to install proper driver and CUDA libraries. To install PaddlePaddle try to read [this](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/getstarted/build_and_install) document. We strongly recommend using [Docker installation](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/getstarted/build_and_install/docker_install_en.rst).

After installation, you can check the version by typing the below command(run into a docker container if using docker: `docker run -it paddlepaddle/paddle:0.10.0rc3 /bin/bash`):

```bash
$ paddle version
PaddlePaddle 0.10.0rc, compiled with
    with_avx: ON
    with_gpu: OFF
    with_double: OFF
    with_python: ON
    with_rdma: OFF
    with_timer: OFF
```

We'll take `demo/word2vec` as an example to introduce distributed training using PaddlePaddle v2 API.

# Command-line arguments

## Starting parameter server

Type the below command to start a parameter server which will wait for trainers to connect:

```bash
$ paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1
```

If you wish to run parameter servers in background, and save a log file, you can type:
```bash
$ stdbuf -oL /usr/bin/nohup paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1 &> pserver.log
```

| param  | required | default | description |
| ------------- | ------------- | ------------- | ------------- |
| port  | required | 7164 | port which parameter server will listen on. If ports_num greater than 1, parameter server will listen on multiple ports for more network throughput |
| ports_num  | required | 1 | total number of ports will listen on  |
| ports_num_for_sparse  | required | 1 | number of ports which serves sparse parameter update  |
| num_gradient_servers  | required | 1 | total number of gradient servers |

## Starting trainer
Type the command below to start the trainer(name the file whatever you want, like "train.py")

```bash
$ python train.py
```

Trainers network need to be connected with parameter servers to finish the job. Trainers need to know port and IPs to locate parameter servers. You can pass arguments to trainers through [environment variables](https://en.wikipedia.org/wiki/Environment_variable) or pass to `paddle.init()` function. Arguments passed to the `paddle.init()` function will overwrite environment variables.

Use environment viriables:

```bash
export PADDLE_INIT_USE_GPU=False
export PADDLE_INIT_TRAINER_COUNT=1
export PADDLE_INIT_PORT=7164
export PADDLE_INIT_PORTS_NUM=1
export PADDLE_INIT_PORTS_NUM_FOR_SPARSE=1
export PADDLE_INIT_NUM_GRADIENT_SERVERS=1
export PADDLE_INIT_TRAINER_ID=0
export PADDLE_INIT_PSERVERS=127.0.0.1
python train.py
```

Pass arguments:

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

| param  | required | default | description |
| ------------- | ------------- | ------------- | ------------- |
| use_gpu  | optional | False | set to "True" to enable GPU training |
| trainer_count  | required | 1 | total count of trainers in the training job |
| port  | required | 7164 | port to connect to parameter server  |
| ports_num  | required | 1 | number of ports for communication |
| ports_num_for_sparse  | required | 1 | number of ports for sparse type caculation |
| num_gradient_servers  | required | 1 | total number of gradient server |
| trainer_id  | required | 0 | ID for every trainer, start from 0 |
| pservers  | required | 127.0.0.1 | list of IPs of parameter servers, separated by "," |

## Prepare Training Dataset

Here's some example code “prepare.py”(located in `demo/word2vec/prepare.py`), it will download public `imikolov` dataset and split it into multiple files according to job parallelism(trainers count). Modify `SPLIT_COUNT` to change the count of output files.

In the real world, we often use `MapReduce` job's output as training data, so there will be lots of files. You can use `mod` to assign training file to trainers:

```python
import os
train_list = []
flist = os.listdir("/train_data/")
for f in flist:
  suffix = int(f.split("-")[1])
  if suffix % TRAINER_COUNT == TRAINER_ID:
    train_list.append(f)
```

Example code `prepare.py` will split training data and testing data into 3 files with digital suffix like `-00000`, `-00001` and`-00002`:

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

When job started, every trainer needs to get it's own part of data. In some distributed systems they will provide a storage service, so the date under that path can be accessed by all the trainer nodes. Without storage service, you must copy the training data to each trainer node.

Different training jobs may have different data format and `reader()` function, developers may need to write different data prepare scripts and `reader()` functions for their job.

## Prepare Training program

We'll create a *workspace* directory on each node, storing your training program, dependencies, mounted or downloaded dataset directory.


Your workspace may looks like:
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

- `mylib.py`: some library functions. This is optional.
- `word_dict.pickle`: dict file for training word embeding.
- `train.py`: training program. Sample code: "api_train_v2_cluster.py"(located in `demo/word2vec/api_train_v2_cluster.py`).
  - ***NOTE:*** You may need to modify the head part of `train.py` when using different cluster platform to retrive configuration environment variables:
  ```python
  cluster_train_file = "./train_data_dir/train/train.txt"
  cluster_test_file = "./test_data_dir/test/test.txt"
  node_id = os.getenv("OMPI_COMM_WORLD_RANK")
  if not node_id:
      raise EnvironmentError("must provied OMPI_COMM_WORLD_RANK")
  ```
- `train_data_dir`: containing training data. Mount from storage service or copy trainning data to here.
- `test_data_dir`: containing testing data.

# Use cluster platforms or cluster management tools

PaddlePaddle supports running jobs on several platforms including:
- [Kubernetes](http://kubernetes.io)
- [OpenMPI](https://www.open-mpi.org)
- [Fabric](http://www.fabfile.org) A cluster management tool. Write scripts to submit jobs or manage the cluster.

We'll introduce cluster job management on these platforms. The examples can be found under [here](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/cluster_train_v2).

These cluster platforms provide API or runtime environment variables for training jobs.

## Cluster Training Using Fabric

### Prepare a demo cluster using kubernetes(optional)

Run `kubectl -f ssh_servers.yaml` under the directory:  `paddle/scripts/cluster_train_v2/fabric/docker_cluster` will launch a demo cluster. Run `kubectl get po -o wide` to get IP addresses of these nodes.

### Launching Cluster Job
`paddle.py` provides automatical scripts to start all PaddlePaddle cluster processes in different nodes. By default, all command line options can set as `paddle.py` command options and `paddle.py` will transparently and automatically set these options to PaddlePaddle lower level processes.

`paddle.py`provides two distinguished command option for easy job launching.

`job_dispatch_package`  set it with local `workspace`directory, it will be dispatched to all nodes set in conf.py. It could be helpful for frequent hacking workspace files, otherwise, frequent multi-nodes workspace deployment could make your crazy.
`job_workspace`  set it with already deployed workspace directory, `paddle.py` will skip dispatch stage to directly launch cluster job with all nodes. It could help to reduce heavy
dispatch latency.

`cluster_train/run.sh` provides command line sample to run `demo/recommendation` cluster job, just modify `job_dispatch_package` and `job_workspace` with your defined directory, then:
```
sh run.sh
```

The cluster Job will start in several seconds.

### Kill Cluster Job
`paddle.py` can capture `Ctrl + C` SIGINT signal to automatically kill all processes launched by it. So just stop `paddle.py` to kill cluster job. You should manually kill the job if the program crashed.

### Check Cluster Training Result
Check log in $workspace/log for details, each node owns same log structure.

`paddle_trainer.INFO`
It provides almost all internal output log for training,  same as local training. Check runtime model convergence here.

`paddle_pserver2.INFO`
It provides parameter server running log, which could help to diagnose distributed error.

`server.log`
It provides stderr and stdout of parameter server process. Check error log if training crashes.

`train.log`
It provides stderr and stdout of trainer process. Check error log if training crashes.

### Check Model Output
After one pass finished, model files will be written in `output` directory in node 0.
`nodefile` in workspace indicates the node id of current cluster job.

## Cluster Training Using OpenMPI

### Prepare a demo OpenMPI cluster using kubernetes(optional)

Run the following command to start a 3-node MPI cluster and one "head" node.

```bash
cd paddle/scripts/cluster_train_v2/openmpi/docker_cluster
kubectl create -f head.yaml
kubectl create -f mpi-nodes.yaml
```

Then you can log in to every OpenMPI node using ssh without input any passwords.

### Launching Cluster Job

Follow the steps to launch a PaddlePaddle training job in OpenMPI cluster:\

```bash
# find out node IP addresses
kubectl get po -o wide
# generate a "machines" file containing node IP addresses
kubectl get po -o wide | grep nodes | awk '{print $6}' > machines
# copy necessary files onto "head" node
scp -i ssh/id_rsa.mpi.pub machines prepare.py train.py start_mpi_train.sh tutorial@[headIP]:~
# login to head node using ssh
ssh -i ssh/id_rsa.mpi.pub tutorial@[headIP]
# --------------- in head node ---------------
# prepare training data
python prepare.py
# copy training data and dict file to MPI nodes
cat machines | xargs -i scp word_dict.pickle train.py start_mpi_train.sh machines {}:/home/tutorial
# creat a directory for storing log files
mpirun -hostfile machines -n 3 mkdir /home/tutorial/logs
# copy training data to every node
scp train.txt-00000 test.txt-00000 [node1IP]:/home/tutorial
scp train.txt-00001 test.txt-00001 [node2IP]:/home/tutorial
scp train.txt-00002 test.txt-00002 [node3IP]:/home/tutorial
# start the job
mpirun -hostfile machines -n 3  /home/tutorial/start_mpi_train.sh
```

## Cluster Training Using Kubernetes

The details can be found [here](../k8s/k8s_cn.md)
