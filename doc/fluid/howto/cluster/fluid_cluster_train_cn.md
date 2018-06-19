# Fluid 分布式版本使用指南
本篇文章将说明如何在PaddlePaddle Fluid版本下进行分布式训练的配置和执行，以及将单机训练脚本改造成支持集群训练的版本

## 准备工作
* 可用的集群

    包含一个或多个计算节点的集群，每一个节点都能够执行PaddlePaddle的训练任务且拥有唯一的IP地址，集群内的所有计算节点可以通过网络相互通信。
* 安装PaddlePaddle Fluid with Distribution版本

    所有的计算节点上均需要按照分布式版本的PaddlePaddle, 在用于GPU等设备的机器上还需要额外安装好相应的驱动程序和CUDA的库。

    **注意：**当前对外提供的PaddlePaddle版本并不支持分布式，需要通过源码重新编译。编译和安装方法参见[编译和安装指南](http://www.paddlepaddle.org/docs/develop/documentation/en/getstarted/build_and_install/index_en.html)。
    cmake编译命令中需要将WITH_DISTRIBUTE设置为ON，下面是一个cmake编译指令示例：
``` bash
cmake .. -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=ON -DWITH_SWIG_PY=ON -DWITH_PYTHON=ON
```

## 更新训练脚本
这里，我们以[Deep Learing 101](http://www.paddlepaddle.org/docs/develop/book/01.fit_a_line/index.html)课程中的第一章 fit a line 为例，描述如何将单机训练脚本改造成支持集群训练的版本。
### 单机训练脚本示例
```python
import paddle.v2 as paddle
import paddle.fluid as fluid

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(x=cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    fluid.io.save_persistables(exe, "./fit_a_line.model/")
    fluid.io.load_persistables(exe, "./fit_a_line.model/")
    for data in train_reader():
        avg_loss_value, = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[avg_cost])

        if avg_loss_value[0] < 10.0:
            exit(0)  # if avg cost less than 10.0, we think our code is good.
exit(1)
```

我们创建了一个简单的全连接神经网络程序，并且通过Fluid的Executor执行了100次迭代,现在我们需要将该单机版本的程序更新为分布式版本的程序。
### 介绍Parameter Server
在非分布式版本的训练脚本中，只存在Trainer一种角色，它不仅处理常规的计算任务，也处理参数相关的计算、保存和优化任务。在分布式版本的训练过程中，由于存在多个Trainer节点进行同样的数据计算任务，因此需要有一个中心化的节点来统一处理参数相关的保存和分配。在PaddlePaddle中，我们称这样的节点为[Parameter Server](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/dist_train/parameter_server.md)

**因此，在分布式的Fluid环境中，我们有两个角色需要创建，分别是Parameter Server和Trainer。**

### 分布式训练
Fliud专门提供了工具[Distributed Transpiler](https://github.com/PaddlePaddle/Paddle/blob/ba65d54d9d3b41cd3c5171b00f476d4e60133ddb/doc/fluid/design/dist_train/distributed_architecture.md#distributed-transpiler)用于将单机版的训练程序转换为分布式版本的训练程序。工具背后的理念是找出程序的优化算子和梯度参数，将他们分隔为两部分，通过send/recv 操作算子进行连接,优化算子和梯度参数可以在优化器的minimize函数的返回值中获取到。
```python
optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)
```
将Distributed Transpiler、优化算子和梯度函数放在一个代码中如下：
```python
... #define the program, cost, and create sgd optimizer

optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost) #get optimize OPs and gradient parameters

t = fluid.DistributeTranspiler() # create the transpiler instance
# slice the program into 2 pieces with optimizer_ops and gradient parameters list, as well as pserver_endpoints, which is a comma separated list of [IP:PORT] and number of trainers
t.transpile(optimize_ops, params_grads, pservers=pserver_endpoints, trainers=2)

... #create executor

# in pserver, run this
#current_endpoint here means current pserver IP:PORT you wish to run on
pserver_prog = t.get_pserver_program(current_endpoint)
pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
exe.run(pserver_startup)
exe.run(pserver_prog)

# in trainer, run this
... # define data reader
exe.run(fluid.default_startup_program())
for pass_id in range(100):
    for data in train_reader():
        exe.run(t.get_trainer_program())
```
### 分布式训练脚本运行说明
分布式任务的运行需要将表格中说明的多个参数进行赋值:

<table>
<thead>
<tr>
<th>参数名</th>
<th> 值类型</th>
<th>说明</th>
<th> 示例</th>
</tr>
</thead>
<tbody>
<tr>
<td>trainer_id </td>
<td> int</td>
<td> 当前训练节点的ID，训练节点ID编号为0 - n-1， n为trainers的值 </td>
<td> 0/1/2/3  </td>
</tr>
<tr>
<td>pservers </td>
<td> str</td>
<td> parameter server 列表 </td>
<td> 127.0.0.1:6710,127.0.0.1:6711 </td>
</tr>
<tr>
<td>trainers </td>
<td>int </td>
<td> 训练节点的总个数，>0的数字 </td>
<td> 4 </td>
</tr>
<tr>
<td> server_endpoint</td>
<td> str </td>
<td> 当前所起的服务节点的IP:PORT </td>
<td> 127.0.0.1:8789 </td>
</tr>
<tr>
<td> training_role</td>
<td>str </td>
<td> 节点角色， TRAINER/PSERVER </td>
<td> PSERVER </td>
</tr>
</tbody>
</table>


**注意：** ```training_role```是用来区分当前所起服务的角色的，用于训练程序中，用户可根据需要自行定义，其他参数为fluid.DistributeTranspiler的transpile函数所需要，需要在调用函数前进行定义，样例如下：

```python
t = fluid.DistributeTranspiler()
t.transpile(
    optimize_ops,
    params_grads,
    trainer_id,
    pservers=pserver,
    trainers=trainers)
if training_role == "PSERVER":
    pserver_prog = t.get_pserver_program(server_endpoint)
    pserver_startup = t.get_startup_program(server_endpoint, pserver_prog)
```

### Demo
完整的demo代码位于Fluid的test目录下的[book](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py)中。

第一步，进入demo代码所在目录：
```bash
cd /paddle/python/paddle/fluid/tests/book
```

第二步，启动Parameter Server：
```bash
PADDLE_PSERVER_PORT=6174 PADDLE_PSERVER_IPS=192.168.1.2 PADDLE_TRAINERS=2 PADDLE_CURRENT_IP=192.168.1.2 PADDLE_TRAINER_ID=1 PADDLE_TRAINING_ROLE=PSERVER python test_fit_a_line.py
```
执行命令后请等待出现提示： ```Server listening on 192.168.1.2:6174 ```, 表示Paramter Server已经正常启动。

第三步，启动Trainer：
```bash
PADDLE_PSERVER_PORT=6174 PADDLE_PSERVER_IPS=192.168.1.3 PADDLE_TRAINERS=2 PADDLE_CURRENT_IPP=192.168.1.3 PADDLE_TRAINER_ID=1 PADDLE_TRAINING_ROLE=TRAINER python test_fit_a_line.py
```
由于我们定义的Trainer的数量是2个，因此需要在另外一个计算节点上再启动一个Trainer。

现在我们就启动了一个包含一个Parameter Server和两个Trainer的分布式训练任务。
