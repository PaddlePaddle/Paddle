############
基本使用概念
############

PaddlePaddle是源于百度的一个深度学习平台。PaddlePaddle为深度学习研究人员提供了丰富的API，可以轻松地完成神经网络配置，模型训练等任务。
这里将介绍PaddlePaddle的基本使用概念，并且展示了如何利用PaddlePaddle来解决一个经典的线性回归问题。
在使用该文档之前，请参考 `安装文档 <../../build_and_install/index_cn.html>`_ 完成PaddlePaddle的安装。


配置网络
============

加载PaddlePaddle
----------------------

在进行网络配置之前，首先需要加载相应的Python库，并进行初始化操作。

..	code-block:: bash

    import paddle.v2 as paddle
    import numpy as np
    paddle.init(use_gpu=False)


搭建神经网络
-----------------------

搭建神经网络就像使用积木搭建宝塔一样。在PaddlePaddle中，layer是我们的积木，而神经网络是我们要搭建的宝塔。我们使用不同的layer进行组合，来搭建神经网络。
宝塔的底端需要坚实的基座来支撑，同样，神经网络也需要一些特定的layer作为输入接口，来完成网络的训练。

例如，我们可以定义如下layer来描述神经网络的输入：

..	code-block:: bash

    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
    y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))

其中x表示输入数据是一个维度为2的稠密向量，y表示输入数据是一个维度为1的稠密向量。

PaddlePaddle支持不同类型的输入数据，主要包括四种类型，和三种序列模式。

四种数据类型：

* dense_vector：稠密的浮点数向量。
* sparse_binary_vector：稀疏的01向量，即大部分值为0，但有值的地方必须为1。
* sparse_float_vector：稀疏的向量，即大部分值为0，但有值的部分可以是任何浮点数。
* integer：整数标签。

三种序列模式：

* SequenceType.NO_SEQUENCE：不是一条序列
* SequenceType.SEQUENCE：是一条时间序列
* SequenceType.SUB_SEQUENCE： 是一条时间序列，且序列的每一个元素还是一个时间序列。

不同的数据类型和序列模式返回的格式不同，列表如下：

+----------------------+---------------------+-----------------------------------+------------------------------------------------+
|                      | NO_SEQUENCE         | SEQUENCE                          |  SUB_SEQUENCE                                  |
+======================+=====================+===================================+================================================+
| dense_vector         | [f, f, ...]         | [[f, ...], [f, ...], ...]         | [[[f, ...], ...], [[f, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_binary_vector | [i, i, ...]         | [[i, ...], [i, ...], ...]         | [[[i, ...], ...], [[i, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_float_vector  | [(i,f), (i,f), ...] | [[(i,f), ...], [(i,f), ...], ...] | [[[(i,f), ...], ...], [[(i,f), ...], ...],...] |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| integer_value        |  i                  | [i, i, ...]                       | [[i, ...], [i, ...], ...]                      |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+

其中，f代表一个浮点数，i代表一个整数。

注意：对sparse_binary_vector和sparse_float_vector，PaddlePaddle存的是有值位置的索引。例如，

- 对一个5维非序列的稀疏01向量 ``[0, 1, 1, 0, 0]`` ，类型是sparse_binary_vector，返回的是 ``[1, 2]`` 。
- 对一个5维非序列的稀疏浮点向量 ``[0, 0.5, 0.7, 0, 0]`` ，类型是sparse_float_vector，返回的是 ``[(1, 0.5), (2, 0.7)]`` 。


在定义输入layer之后，我们可以使用其他layer进行组合。在组合时，需要指定layer的输入来源。

例如，我们可以定义如下的layer组合：

..	code-block:: bash

    y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
    cost = paddle.layer.square_error_cost(input=y_predict, label=y)

其中，x与y为之前描述的输入层；而y_predict是接收x作为输入，接上一个全连接层；cost接收y_predict与y作为输入，接上平方误差层。

最后一层cost中记录了神经网络的所有拓扑结构，通过组合不同的layer，我们即可完成神经网络的搭建。


训练模型
============

在完成神经网络的搭建之后，我们首先需要根据神经网络结构来创建所需要优化的parameters，并创建optimizer。
之后，我们可以创建trainer来对网络进行训练。

..	code-block:: bash

    parameters = paddle.parameters.create(cost)
    optimizer = paddle.optimizer.Momentum(momentum=0)
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer)

其中，trainer接收三个参数，包括神经网络拓扑结构、神经网络参数以及迭代方程。

在搭建神经网络的过程中，我们仅仅对神经网络的输入进行了描述。而trainer需要读取训练数据进行训练，PaddlePaddle中通过reader来加载数据。

..	code-block:: bash

    # define training dataset reader
    def train_reader():
        train_x = np.array([[1, 1], [1, 2], [3, 4], [5, 2]])
        train_y = np.array([[-2], [-3], [-7], [-7]])
        def reader():
            for i in xrange(train_y.shape[0]):
                yield train_x[i], train_y[i]
        return reader

最终我们可以调用trainer的train方法启动训练：

..	code-block:: bash

    # define feeding map
    feeding = {'x': 0, 'y': 1}

    # event_handler to print training info
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                print "Pass %d, Batch %d, Cost %f" % (
                    event.pass_id, event.batch_id, event.cost)
    # training
    trainer.train(
        reader=paddle.batch(train_reader(), batch_size=1),
        feeding=feeding,
        event_handler=event_handler,
        num_passes=100)

关于PaddlePaddle的更多使用方法请参考 `进阶指南 <../../howto/index_cn.html>`_。

线性回归完整示例
==============

下面给出在三维空间中使用线性回归拟合一条直线的例子：

..  literalinclude:: src/train.py
    :linenos:

使用以上训练好的模型进行预测，取其中一个模型params_pass_90.tar，输入需要预测的向量组，然后打印输出：

..  literalinclude:: src/infer.py
    :linenos:

有关线性回归的实际应用，可以参考PaddlePaddle book的 `第一章节 <http://book.paddlepaddle.org/index.html>`_。
