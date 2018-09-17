..  _cluster_quick_start:

分布式训练快速开始
==================

准备工作
--------

在本篇文章中，我们将会在介绍如何快速在一个集群中启动一个 PaddlePaddle
的分布式训练任务，在开始之前，请按如下步骤做些准备工作：

1. 准备一个网络连通的训练集群，在本文中我们使用4个训练节点使用 ``*.paddlepaddle.com``
   来表示节点的主机名称，您可以根据实际情况修改它。

2. 在开始之前确保已经阅读过 :ref:`how_to_install`
   并且可以在集群的所有节点上可以正常运行 PaddlePaddle。

样例代码
-------

下面使用一个非常简单的限行回归模型作为样例来解释如何启动一个包含2个 pserver 节点以及
2个 trainer 节点的分布式训练任务，您可以将本段代码保存为 ``dist_train.py``

.. code:: python

    import os
    import paddle
    import paddle.fluid as fluid

    # train reader
    BATCH_SIZE = 20
    EPOCH_NUM = 30
    BATCH_SIZE = 8

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    def train():
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)

        loss = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_loss = fluid.layers.mean(loss)
        opt = fluid.optimizer.SGD(learning_rate=0.001)
        opt.minimize(avg_loss)

        place = fluid.CPUPlace()
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)

        # fetch distributed training environment setting
        training_role = os.getenv("PADDLE_TRAINING_ROLE", None)
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port

        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id = trainer_id,
            pservers = pserver_endpoints,
            trainers = trainers)

        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
            exe.run(startup_prog)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            trainer_prog = t.get_trainer_program()
            exe.run(fluid.default_startup_program())

            for epoch in range(EPOCH_NUM):
                for batch_id, batch_data in enumerate(train_reader()):
                    avg_loss_value, = exe.run(trainer_prog,
                                          feed=feeder.feed(batch_data),
                                          fetch_list=[avg_loss])
                    if (batch_id + 1) % 10 == 0:
                        print("Epoch: {0}, Batch: {1}, loss: {2}".format(
                            epoch, batch_id, avg_loss_value[0]))
            # destory the resource of current trainer node in pserver server node
            exe.close()
        else:
            raise AssertionError("PADDLE_TRAINING_ROLE should be one of [TRAINER, PSERVER]")

    train()

环境变量说明
-----------

在启动分布式训练任务时，使用不同的环境变量来表示不同的节点角色，具体如下：

.. list-table::
  :header-rows: 1

  * - 环境变量
    - 数据类型
    - 样例
    - 描述
  * - :code:`PADDLE_TRAINING_ROLE`
    - str
    - :code:`PSERVER,TRAINER`
    - 当前训练节点角色
  * - :code:`PADDLE_PSERVER_IPS`
    - str
    - :code:`ps0.paddlepaddle.com,ps1.paddlepaddle.com`
    - 分布式训练任务中所有 pserver 节点的 IP 地址或 hostname, 使用","分隔
  * - :code:`PADDLE_PSERVER_PORT`
    - int
    - 6174
    - pserver 进程监听的端口
  * - :code:`PADDLE_TRAINERS`
    - int
    - 2
    - 分布式训练任务中 trainer 节点的数量
  * - :code:`PADDLE_CURRENT_IP`
    - str
    - ps0.paddlepaddle.com
    - 当前 pserver 节点的 IP 地址或 hostname
  * - :code:`PADDLE_TRAINER_ID`
    - str 
    - 0
    - 当前 trainer 节点的 ID (唯一)， 取值范围为 [0, PADDLE_TRAINERS)


分布式训练相关 API
------------------

DistributeTranspiler
~~~~~~~~~~~~~~~~~~~~~~

基于 pserver-trainer 架构的的分布式训练任务分为两种角色： Parameter Server(pserver) 以及 trainer, 
在 Fluid 中，用户只需配置单机训练所需要的网络配置, ``DistributeTranspiler`` 模块会自动地根据
当前训练节点的角色将用户配置的单机网路配置改写成 pserver 和 trainer 需要运行的网络配置:

.. code:: python

    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id = trainer_id,                   
        pservers = pserver_endpoints,    
        trainers = trainers)
    if PADDLE_TRAINING_ROLE == "TRAINER":
        # fetch the pserver program and execute it
        trainer_prog = t.get_trainer_program()
        ...

    elif PADDLE_TRAINER_ROLE == "PSERVER":
        # fetch the trainer program and execute it
        pserver_prog = t.get_pserver_program(current_endpoint) 
        ...

exe.close()
~~~~~~~~~~~~~~

pserver 节点中会保存所有 trainer 节点的状态信息，在 trainer结束训练时需要调用 ``exe.close()``
通知所有 PServer 节点释放当前 Trainer 节点的资源:

.. code:: python

    exe = fluid.Executor(fluid.CPUPlace())
    # training process ...
    exe.close() # notify PServer to destory the resource

启动分布式训练任务
--------------------

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
