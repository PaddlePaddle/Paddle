############
基本使用概念
############

PaddlePaddle是一个深度学习框架，支持单机模式和多机模式。

单机模式用命令 ``paddle train`` 可以启动一个trainer进程，单机训练通常只包括一个trainer进程。如果数据规模比较大，希望加速训练，可以启动分布式作业。一个分布式作业里包括若干trainer进程和若干Parameter Server（或称pserver）进程。用命令 ``paddle pserver`` 可以启动 pserver 进程，pserver进程用于协调多个trainer进程之间的通信。

本文首先介绍trainer进程中的一些使用概念，然后介绍pserver进程中概念。

..    contents::

系统框图
========

下图描述了用户使用框图，PaddlePaddle的trainer进程里内嵌了Python解释器，trainer进程可以利用这个解释器执行Python脚本，Python脚本里定义了模型配置、训练算法、以及数据读取函数。其中，数据读取程序往往定义在一个单独Python脚本文件里，被称为数据提供器（DataProvider），通常是一个Python函数。模型配置、训练算法通常定义在另一单独Python文件中, 称为训练配置文件。下面将分别介绍这两部分。

..    graphviz:: 

    digraph pp_process {
        rankdir=LR;
        config_file [label="用户神经网络配置"];
        subgraph cluster_pp {
            style=filled;
            color=lightgrey;
            node [style=filled, color=white, shape=box];
            label = "PaddlePaddle C++";
            py [label="Python解释器"];
        }
        data_provider [label="用户数据解析"];
        config_file -> py;
        py -> data_provider [dir="back"];
    }

数据提供器
==========

DataProvider是PaddlePaddle系统的数据提供器，将用户的原始数据转换成系统可以识别的数据类型。每当系统需要新的数据训练时, trainer进程会调用DataProvider函数返回数据。当所有数据读取完一轮后，DataProvider返回空数据，通知系统一轮数据读取结束，并且系统每一轮训练开始时会重置DataProvider。需要注意的是，DataProvider是被系统调用，而不是新数据驱动系统，一些随机化噪声添加都应该在DataProvider中完成。

在不同的应用里，训练数据的格式往往各不相同。因此，为了用户能够灵活的处理数据，我们提供了Python处理数据的接口，称为 ``PyDataProvider`` 。在 ``PyDataProvider`` 中，系统C++模块接管了shuffle、处理batch、GPU和CPU通信、双缓冲、异步读取等问题，一些情况下(如：``min_pool_size=0``)需要Python接口里处理shuffle，可以参考 :ref:`api_pydataprovider2` 继续深入了解。


训练配置文件
============

训练配置文件主要包括数据源、优化算法、网络结构配置三部分。 其中数据源配置与DataProvider的关系是：DataProvider里定义数据读取函数，训练配置文件的数据源配置中指定DataProvider文件名字、生成数据函数接口，请不要混淆。

一个简单的训练配置文件为：

..  literalinclude:: src/trainer_config.py
    :linenos:

文件开头 ``from paddle.trainer_config_helpers import *`` ，是因为PaddlePaddle配置文件与C++模块通信的最基础协议是protobuf，为了避免用户直接写复杂的protobuf string，我们为用户定以Python接口来配置网络，该Python代码可以生成protobuf包，这就是 :ref:`api_trainer_config` 的作用。因此，在文件的开始，需要import这些函数。 这个包里面包含了模型配置需要的各个模块。

下面分别介绍数据源配置、优化算法配置、网络结构配置这三部分该概念。

数据源配置
----------

使用 ``PyDataProvider2`` 的函数 ``define_py_data_sources2`` 配置数据源。``define_py_data_sources2`` 里通过train_list和test_list指定是训练文件列表和测试文件列表。 如果传入字符串的话，是指一个数据列表文件。这个数据列表文件中包含的是每一个训练或者测试文件的路径。如果传入一个list的话，则会默认生成一个list文件，再传入给train.list或者test.list。

``module`` 和 ``obj`` 指定了DataProvider的文件名和返回数据的函数名。更详细的使用，请参考 :ref:`api_pydataprovider2` 。

优化算法配置
------------

通过 :ref:`api_trainer_config_helpers_optimizers_settings` 接口设置神经网络所使用的训练参数和 :ref:`api_trainer_config_helpers_optimizers` ，包括学习率、batch_size、优化算法、正则方法等，具体的使用方法请参考 :ref:`api_trainer_config_helpers_optimizers_settings` 文档。

网络结构配置
------------

神经网络配置主要包括网络连接、激活函数、损失函数、评估器。

- 网络连接： 主要由Layer组成，每个Layer返回的都是一个 ``LayerOutput`` 对象，Layer里面可以定义参数属性、激活类型等。

  为了更灵活的配置，PaddlePaddle提供了基于 Projection 或者 Operator 的配置，这两个需要与 ``mixed_layer`` 配合使用。这里简单介绍Layer、Projection、Operator的概念:

  - Layer: 神经网络的某一层，可以有可学习的参数，一般是封装了许多复杂操作的集合。
  - Projection：需要与 ``mixed_layer`` 配合使用，含可学习参数。
  - Operator： 需要与 ``mixed_layer`` 配合使用，不含可学习参数，输入全是其他Layer的输出。

 
  这个配置文件网络由 ``data_layer`` 、 ``simple_img_conv_pool`` 、 ``fc_layer`` 组成。

  - :ref:`api_trainer_config_helpers_layers_data_layer`  ： 通常每个配置文件都会包括 ``data_layer`` ，定义输入数据大小。
  - :ref:`api_trainer_config_helpers_network_simple_img_conv_pool` ：是一个组合层，包括了图像的卷积 (convolution)和池化(pooling)。
  - :ref:`api_trainer_config_helpers_layers_fc_layer` ：全连接层，激活函数为Softmax，这里也可叫分类层。

- 损失函数和评估器：损失函数即为网络的优化目标，评估器可以评价模型结果。

  PaddlePaddle包括很多损失函数和评估起，详细可以参考 :ref:`api_trainer_config_helpers_layers_cost_layers` 和 :ref:`api_trainer_config_helpers_evaluators` 。这里 ``classification_cost`` 默认使用多类交叉熵损失函数和分类错误率统计评估器。
  
- ``outputs``: 标记网络输出的函数为 ``outputs`` 。

  训练阶段，网络的输出为神经网络的优化目标；预测阶段，网络的输出也可通过 ``outputs`` 标记。


这里对 ``mixed_layer`` 稍做详细说明， 该Layer将多个输入(Projection 或 Operator)累加求和，具体计算是通过内部的 Projection 和 Operator 完成，然后加 Bias 和 activation 操作，

例如，和 ``fc_layer`` 同样功能的 ``mixed_layer`` 是:

..    code-block:: python
   
       data = data_layer(name='data', size=200)
       with mixed_layer(size=200) as out:
           out += full_matrix_projection(input=data)

PaddlePaddle 可以使用 ``mixed layer`` 配置出非常复杂的网络，甚至可以直接配置一个完整的LSTM。用户可以参考 :ref:`api_trainer_config_helpers_layers_mixed_layer` 的相关文档进行配置。


分布式训练
==========

PaddlePaddle多机采用经典的 Parameter Server 架构对多个节点的 trainer 进行同步。多机训练的经典拓扑结构如下\:

..    graphviz:: src/pserver_topology.dot

图中每个灰色方块是一台机器，在每个机器中，先使用命令 ``paddle pserver`` 启动一个pserver进程，并指定端口号，可能的参数是\:

..    code-block:: bash

    paddle pserver --port=5000 --num_gradient_servers=4 --tcp_rdma='tcp' --nics='eth0'

* ``--port=5000`` : 指定 pserver 进程端口是 5000 。
* ``--gradient_servers=4`` : 有四个训练进程(PaddlePaddle 将 trainer 也称作 GradientServer ，因为其为负责提供Gradient) 。
* ``--tcp_rdma='tcp' --nics=`eth0```: 指定以太网类型为TCP网络，指定网络接口名字为eth0。

启动之后 pserver 进程之后，需要启动 trainer 训练进程，在各个机器上运行如下命令\:

..    code-block:: bash

    paddle train --port=5000 --pservers=192.168.100.101,192.168.100.102,192.168.100.103,192.168.100.104 --config=...

对于简单的多机协同训练使用上述方式即可。另外，pserver/train 通常在高级情况下，还需要设置下面两个参数\：

* --ports_num\: 一个 pserver 进程共绑定多少个端口用来做稠密更新，默认是1。
* --ports_num_for_sparse\: 一个pserver进程共绑定多少端口用来做稀疏更新，默认是0。

使用手工指定端口数量，是因为Paddle的网络通信中，使用了 int32 作为消息长度，比较容易在大模型下溢出。所以，在 pserver 进程中可以启动多个子线程去接受 trainer 的数据，这样单个子线程的长度就不会溢出了。但是这个值不可以调的过大，因为增加这个值，对性能尤其是内存占用有一定的开销，另外稀疏更新的端口如果太大的话，很容易导致某一个参数服务器没有分配到任何参数。
