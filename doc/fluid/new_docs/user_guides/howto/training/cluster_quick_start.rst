..  _cluster_quick_start:

分布式训练快速开始
==================

准备工作
--------

在本篇文章中，我们将会在介绍如何快速在一个集群中启动一个 PaddlePaddle
的分布式训练任务，在开始之前，请按如下步骤做些准备工作：

1. 准备一个至少4个节点的集群，并且保证网络可以联通，在本文中我们使用
   ``*.paddlepaddle.com`` 来表示每个节点的主机名称，您可以根据集群的实际情况来修改它。

2. 在开始之前确保已经阅读过 :ref:`how_to_install`
   并且可以在集群的所有节点上可以正常运行 PaddlePaddle。

启动集群训练任务
----------------

在启动集群训练脚本时，需要在不同的节点上指定不同的环境变量，具体如下：

+-----------------+-----------------+-----------------+---------------------+
| 环境变量        | 数据类型        | 样例            | 描述                |
+=================+=================+=================+=====================+
| PADDLE_TRAINING | str             | PSERVER,TRAINER | 训练节点的角色      |
| _ROLE           |                 |                 |                     |
+-----------------+-----------------+-----------------+---------------------+
| PADDLE_PSERVER_ | str             | ps0.paddlepaddl | 所有 pserver        |
| IPS             |                 | e.com,ps1.paddl | 节点的 IP           |
|                 |                 | epaddle.com…    | 地址或              |
|                 |                 |                 | hostname,           |
|                 |                 |                 | 用“,”分隔           |
+-----------------+-----------------+-----------------+---------------------+
| PADDLE_PSERVER_ | int             | 6174            | pserver             |
| PORT            |                 |                 | 节点监听的端口      |
+-----------------+-----------------+-----------------+---------------------+
| PADDLE_TRAINERS | int             | 2               | 训练任务中          |
|                 |                 |                 | trainer             |
|                 |                 |                 | 节点的数量          |
+-----------------+-----------------+-----------------+---------------------+
| PADDLE_CURRENT_ | str             | ps0.paddlepaddl | 当前 pserver        |
| IP              |                 | e.com           | 节点的 IP           |
|                 |                 |                 | 地址或 hostanme     |
+-----------------+-----------------+-----------------+---------------------+
| PADDLE_TRAINER_ | int             | 0               | 当前 trainer        |
| ID              |                 |                 | 节点的唯一 ID,      |
|                 |                 |                 | 取值范围为从0开始到 |
|                 |                 |                 | PADDLE_TRAINERS-1   |
+-----------------+-----------------+-----------------+---------------------+

样例代码
~~~~~~~~

将下面程序代码保存为 ``fluid_dist.py``

.. code:: python

   import paddle
   import paddle.fluid as fluid
   import contextlib
   import numpy
   import unittest

   # train reader
   BATCH_SIZE = 20

   train_reader = paddle.batch(
       paddle.reader.shuffle(
           paddle.dataset.uci_housing.train(), buf_size=500),
       batch_size=BATCH_SIZE)

   test_reader = paddle.batch(
       paddle.reader.shuffle(
           paddle.dataset.uci_housing.test(), buf_size=500),
       batch_size=BATCH_SIZE)


   def train_program():
       y = fluid.layers.data(name='y', shape=[1], dtype='float32')
       x = fluid.layers.data(name='x', shape=[13], dtype='float32')
       y_predict = fluid.layers.fc(input=x, size=1, act=None)

       loss = fluid.layers.square_error_cost(input=y_predict, label=y)
       avg_loss = fluid.layers.mean(loss)

       return avg_loss

   def optimizer_func():
       return fluid.optimizer.SGD(learning_rate=0.001)

   def train(use_cuda, train_program):
       place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

       trainer = fluid.Trainer(
           train_func=train_program, place=place, optimizer_func=optimizer_func)

       def event_handler(event):
           if isinstance(event, fluid.EndStepEvent):
               if event.step == 10:
                   test_metrics = trainer.test(
                       reader=test_reader, feed_order=['x', 'y'])
                   print("step {0}, loss: {1}".format(event.step, test_metrics))
                   trainer.stop()

       trainer.train(
           reader=train_reader,
           num_epochs=100,
           event_handler=event_handler,
           feed_order=['x', 'y'])

   train(False, train_program)

启动trainer节点和pserver节点
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 启动节点
     - 启动命令
     - 说明
   * - ps0.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=PSERVER PADDLE_CURRENT_IP=ps0.paddlepaddle.com PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动 pserver 节点
   * - ps1.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=PSERVER PADDLE_CURRENT_IP=ps1.paddlepaddle.com PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动 pserver 节点
   * - trainer0.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_TRAINER_ID=0 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动第0号 trainer 节点
   * - trainer1.paddlepaddle.com
     - :code:`PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_IPS=ps0.paddlepaddle.com,ps1.paddlepaddle.com PADDLE_TRAINERS=2 PADDLE_TRAINER_ID=1 PADDLE_PSERVER_PORT=6174 python fluid_dist.py`
     - 启动第1号 trainer 节点

**注意**

-  需要先启动pserver节点再启动trainer节点
-  看到trainer节点输出如下日志表示训练任务执行正确

   .. code:: bash

      step 10, loss: [258.2326202392578]
