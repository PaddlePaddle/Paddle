# 启动参数说明

下面以`doc/howto/cluster/src/word2vec`中的代码作为实例，介绍使用PaddlePaddle v2 API完成分布式训练。

## 启动参数服务器

执行以下的命令启动一个参数服务器并等待和计算节点的数据交互

```bash
$ paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1
```

如果希望可以在后台运行pserver程序，并保存输出到一个日志文件，可以运行：

```bash
$ stdbuf -oL /usr/bin/nohup paddle pserver --port=7164 --ports_num=1 --ports_num_for_sparse=1 --num_gradient_servers=1 &> pserver.log
```

参数说明

- port：**必选，默认7164**，pserver监听的起始端口，根据ports_num决定总端口个数，从起始端口监听多个端口用于通信
- ports_num：**必选，默认1**，监听的端口个数
- ports_num_for_sparse：**必选，默认0**，用于稀疏类型参数通信的端口个数
- num_gradient_servers：**必选，默认1**，当前训练任务pserver总数

## 启动计算节点

执行以下命令启动使用python编写的trainer程序（文件名为任意文件名，如train.py）

```bash
$ python train.py
```

trainer需要和pserver保持网络联通以完成训练。trainer启动需要传入端口、pserver地址等参数使trainer可以正确连接到pserver。这些参数可以通过[环境变量](https://zh.wikipedia.org/wiki/环境变量)或编写程序时`paddle.init()`中传入参数。如果同时使用`paddle.init()`参数和环境变量，将会优先使用`paddle.init()`中传入的参数。

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

参数说明

- use_gpu： **可选，默认False**，是否启用GPU训练
- trainer_count：**必选，默认1**，当前trainer的线程数目
- port：**必选，默认7164**，连接到pserver的端口
- ports_num：**必选，默认1**，连接到pserver的端口个数
- ports_num_for_sparse：**必选，默认0**，和pserver之间用于稀疏类型参数通信的端口个数
- num_gradient_servers：**必选，默认1**，当前训练任务trainer总数
- trainer_id：**必选，默认0**，每个trainer的唯一ID，从0开始的整数
- pservers：**必选，默认127.0.0.1**，当前训练任务启动的pserver的IP列表，多个IP使用“,”隔开

```python
trainer = paddle.trainer.SGD(..., is_local=False)
```

参数说明

- is_local: **必选, 默认True**, 是否使用PServer更新参数

## 准备数据集

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

在进行分布式训练时，每个trainer进程需要能够读取属于自己的一份数据。在一些分布式系统中，系统会提供一个分布式存储服务，这样保存在分布式存储中的数据可以被集群中的每个节点读取到。如果不使用分布式存储，则需要手动拷贝属于每个trainer节点的训练数据到对应的节点上。

对于不同的训练任务，训练数据格式和训练程序的`reader()`会大不相同，所以开发者需要根据自己训练任务的实际场景完成训练数据的分割和`reader()`的编写。

## 准备训练程序

我们会对每个训练任务都会在每个节点上创建一个工作空间（workspace），其中包含了用户的训练程序、程序依赖、挂载或下载的训练数据分片。

最后，工作空间应如下所示：

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

- `my_lib.py`：会被`train.py`调用的一些用户定义的库函数，比如PIL库等。
- `word_dict.pickle`：在`train.py`中会使用到的字典数据文件。
- `train.py`：训练程序，代码参考[api_train_v2_cluster.py](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/howto/usage/cluster/src/word2vec/api_train_v2_cluster.py)。***注意：*** 对于本样例代码，在使用不同的分布式计算平台时，您可能需要修改`train.py`开头的部分（如下），以便获得训练数据的位置和获取环境变量配置：

  ```python
  cluster_train_file = "./train_data_dir/train/train.txt"
  cluster_test_file = "./test_data_dir/test/test.txt"
  node_id = os.getenv("OMPI_COMM_WORLD_RANK")
  if not node_id:
      raise EnvironmentError("must provied OMPI_COMM_WORLD_RANK")
  ```

- `train_data_dir`：包含训练数据的目录，可以是从分布式存储挂载过来的，也可以是在任务启动前下载到本地的。
- `test_data_dir`：包含测试数据集的目录。

## 异步 SGD 更新

我们可以通过设置 `optimize` 的参数使之支持异步SGD更新。
例如，设置 `AdaGrad` optimize 的 `is_async` 和 `async_lagged_grad_discard_ratio` 参数：

```python
adagrad = paddle.optimizer.AdaGrad(
    is_async=True,
    async_lagged_grad_discard_ratio=1.6,
    learning_rate=3e-3,
    regularization=paddle.optimizer.L2Regularization(8e-4))
```

- `is_async`: 是否为异步SGD更新模式。
- `async_lagged_grad_discard_ratio`: 异步SGD更新的步长控制，接收到足够的gradient(
  `async_lagged_grad_discard_ratio * num_gradient_servers`)之后，后面的gradient
  将会被抛弃。
