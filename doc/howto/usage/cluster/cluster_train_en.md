# PaddlePaddle Distributed Training

## Introduction

In this article, we'll explain how to run distributed training jobs with PaddlePaddle on different types of clusters. The diagram below shows the main architecture of a distributed trainning job:

<img src="https://user-images.githubusercontent.com/13348433/31772146-41523d84-b511-11e7-8a12-a69fd136c283.png" width="500">

- Data shard: training data will be split into multiple partitions, trainers use the partitions of the whole dataset to do the training job.
- Trainer: each trainer reads the data shard, and train the neural network. Then the trainer will upload calculated "gradients" to parameter servers, and wait for parameters to be optimized on the parameter server side. When that finishes, the trainer download optimized parameters and continues its training.
- Parameter server: every parameter server stores part of the whole neural network model data. They will do optimization calculations when gradients are uploaded from trainers, and then send updated parameters to trainers.

PaddlePaddle can support both synchronize stochastic gradient descent (SGD) and asynchronous SGD.

When training with synchronize SGD, PaddlePaddle uses an internal "synchronize barrier" which makes gradients update and parameter download in strict order. On the other hand, asynchronous SGD won't wait for all trainers to finish upload at a single step, this will increase the parallelism of distributed training: parameter servers do not depend on each other, they'll do parameter optimization concurrently. Parameter servers will not wait for trainers, so trainers will also do their work concurrently. But asynchronous SGD will introduce more randomness and noises in the gradient.

## Preparations
1. Prepare your computer cluster. It's normally a bunch of Linux servers connected by LAN. Each server will be assigned a unique IP address. The computers in the cluster can be called "nodes".
2. Install PaddlePaddle on every node. If you are going to take advantage of GPU cards, you'll also need to install proper driver and CUDA libraries. To install PaddlePaddle please read [this build and install](http://www.paddlepaddle.org/docs/develop/documentation/en/getstarted/build_and_install/index_en.html) document. We strongly recommend using [Docker installation](http://www.paddlepaddle.org/docs/develop/documentation/en/getstarted/build_and_install/docker_install_en.html).

After installation, you can check the version by typing the below command (run a docker container  if using docker: `docker run -it paddlepaddle/paddle:[tag] /bin/bash`):

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

We'll take `doc/howto/usage/cluster/src/word2vec` as an example to introduce distributed training using PaddlePaddle v2 API.

## Command-line arguments

### Starting parameter server

Type the below command to start a parameter server which will wait for trainers to connect:

```bash
$ paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1
```

If you wish to run parameter servers in background, and save a log file, you can type:
```bash
$ stdbuf -oL /usr/bin/nohup paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1 &> pserver.log
```

Parameter Description

- port: **required, default 7164**, port which parameter server will listen on. If ports_num greater than 1, parameter server will listen on multiple ports for more network throughput.
- ports_num: **required, default 1**, total number of ports will listen on.
- ports_num_for_sparse: **required, default 1**, number of ports which serves sparse parameter update.
- num_gradient_servers: **required, default 1**, total number of gradient servers.

### Starting trainer
Type the command below to start the trainer(name the file whatever you want, like "train.py")

```bash
$ python train.py
```

Trainers' network need to be connected with parameter servers' network to finish the job. Trainers need to know port and IPs to locate parameter servers. You can pass arguments to trainers through [environment variables](https://en.wikipedia.org/wiki/Environment_variable) or pass to `paddle.init()` function. Arguments passed to the `paddle.init()` function will overwrite environment variables.

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

Parameter Description

- use_gpu: **optional, default False**, set to "True" to enable GPU training.
- trainer_count: **required, default 1**, total count of trainers in the training job.
- port: **required, default 7164**, port to connect to parameter server.
- ports_num: **required, default 1**, number of ports for communication.
- ports_num_for_sparse: **required, default 1**, number of ports for sparse type caculation.
- num_gradient_servers: **required, default 1**, total number of gradient server.
- trainer_id: **required, default 0**, ID for every trainer, start from 0.
- pservers: **required, default 127.0.0.1**, list of IPs of parameter servers, separated by ",".

### Prepare Training Dataset

Here's some example code [prepare.py](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/howto/usage/cluster/src/word2vec/prepare.py), it will download public `imikolov` dataset and split it into multiple files according to job parallelism(trainers count). Modify `SPLIT_COUNT` at the begining of `prepare.py` to change the count of output files.

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

When job started, every trainer needs to get it's own part of data. In some distributed systems a storage service will be provided, so the date under that path can be accessed by all the trainer nodes. Without the storage service, you must copy the training data to each trainer node.

Different training jobs may have different data format and `reader()` function, developers may need to write different data prepare scripts and `reader()` functions for their job.

### Prepare Training program

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

- `my_lib.py`: user defined libraries, like PIL libs. This is optional.
- `word_dict.pickle`: dict file for training word embeding.
- `train.py`: training program. Sample code: [api_train_v2_cluster.py](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/howto/usage/cluster/src/word2vec/api_train_v2_cluster.py). ***NOTE:*** You may need to modify the head part of `train.py` when using different cluster platform to retrive configuration environment variables:

  ```python
  cluster_train_file = "./train_data_dir/train/train.txt"
  cluster_test_file = "./test_data_dir/test/test.txt"
  node_id = os.getenv("OMPI_COMM_WORLD_RANK")
  if not node_id:
      raise EnvironmentError("must provied OMPI_COMM_WORLD_RANK")
  ```

- `train_data_dir`: containing training data. Mount from storage service or copy trainning data to here.
- `test_data_dir`: containing testing data.

## Use cluster platforms or cluster management tools

PaddlePaddle supports running jobs on several platforms including:
- [Kubernetes](http://kubernetes.io) open-source system for automating deployment, scaling, and management of containerized applications from Google.
- [OpenMPI](https://www.open-mpi.org) Mature high performance parallel computing framework.
- [Fabric](http://www.fabfile.org) A cluster management tool. Write scripts to submit jobs or manage the cluster.

We'll introduce cluster job management on these platforms. The examples can be found under [cluster_train_v2](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/scripts/cluster_train_v2).

These cluster platforms provide API or environment variables for training processes, when the job is dispatched to different nodes. Like node ID, IP or total number of nodes etc.

## Use different clusters

  - [fabric](fabric_en.md)
  - [openmpi](openmpi_en.md)
  - [kubernetes](k8s_en.md)
  - kubernetes distributed
  - [kubernetes on AWS](k8s_aws_en.md)
