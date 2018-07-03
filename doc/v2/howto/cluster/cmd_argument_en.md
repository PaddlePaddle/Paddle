# Command-line arguments

We'll take `doc/howto/cluster/src/word2vec` as an example to introduce distributed training using PaddlePaddle v2 API.

## Starting parameter server

Type the below command to start a parameter server which will wait for trainers to connect:

```bash
$ paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1 --nics=eth0
```

If you wish to run parameter servers in background, and save a log file, you can type:

```bash
$ stdbuf -oL /usr/bin/nohup paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1 --nics=eth0 &> pserver.log &
```

Parameter Description

- port: **required, default 7164**, port which parameter server will listen on. If ports_num greater than 1, parameter server will listen on multiple ports for more network throughput.
- ports_num: **required, default 1**, total number of ports will listen on.
- ports_num_for_sparse: **required, default 0**, number of ports which serves sparse parameter update.
- num_gradient_servers: **required, default 1**, total number of gradient servers.
- nics: **optional, default xgbe0,xgbe1**, network device name which paramter server will listen on.

## Starting trainer

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
- trainer_count: **required, default 1**, number of threads in current trainer.
- port: **required, default 7164**, port to connect to parameter server.
- ports_num: **required, default 1**, number of ports for communication.
- ports_num_for_sparse: **required, default 0**, number of ports for sparse type caculation.
- num_gradient_servers: **required, default 1**, number of trainers in current job.
- trainer_id: **required, default 0**, ID for every trainer, start from 0.
- pservers: **required, default 127.0.0.1**, list of IPs of parameter servers, separated by ",".

```python
trainer = paddle.trainer.SGD(..., is_local=False)
```

Parameter Description

- is_local: **required, default True**, whether update parameters by PServer.

## Prepare Training Dataset

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

```bash
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

## Prepare Training program

We'll create a *workspace* directory on each node, storing your training program, dependencies, mounted or downloaded dataset directory.

Your workspace may looks like:

```bash
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

## Async SGD Update

We can set some parameters of the optimizer to make it support async SGD update.
For example, we can set the `is_async` and `async_lagged_grad_discard_ratio` of the `AdaGrad` optimizer:

```python
adagrad = paddle.optimizer.AdaGrad(
    is_async=True,
    async_lagged_grad_discard_ratio=1.6,
    learning_rate=3e-3,
    regularization=paddle.optimizer.L2Regularization(8e-4))
```

- `is_async`: Is Async-SGD or not.
- `async_lagged_grad_discard_ratio`: For async SGD gradient commit control.
  when `async_lagged_grad_discard_ratio * num_gradient_servers` commit passed,
  current async gradient will be discard silently.
