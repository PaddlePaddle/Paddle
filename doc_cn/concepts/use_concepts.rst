#########################
PaddlePaddle 基本使用概念
#########################

PaddlePaddle是一个深度学习框架，同时支持单机和多机模式的系统。命令 ``paddle train`` 可启动单机模式的进程，我们称之为 ``trainer`` 进程。单机所有设备使用均在单机进程内调度完成。多机模式除了需要启动trainer进程外，还需要通过命令 ``paddle pserver`` 启动多机参数服务器进程, 我们称之为   ``pserver`` 进程。该进程负责多个单机进程间的通信，进而充分利用集群的计算资源。 PaddlePaddle同时以 ``swig api`` 的形式，提供训练结果模型预测的方法和自定义训练流程。

下面我们会介绍trainer进程中的一些概念，这些概念会对如何使用PaddlePaddle有一定的帮助。 了解这些概念的前提是，读者已经了解 `基本的神经网络/机器学习原理和概念 <nn.html>`_ 。同时，如果想要了解PaddlePaddle实现中的一些概念，请参考 `PaddlePaddle 编程中的基本概念 <program_concepts.html>`_ 。

..	contents::

系统模块
========

``trainer`` 进程内嵌了一个 ``python`` 解释器， 这个 ``python`` 解释器负责解析用户定义的神经网络配置；解析输入数据流，并将数据传入给 ``trainer`` 系统。

..	graphviz:: 

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

所以，单机训练 ``trainer`` 进程对用户的主要接口语言为Python。用户需要配置文件主要有两个：数据流提供器 ``DataProvider`` 和模型配置 ``TrainerConfig``  。


DataProvider
============

DataProvider是 ``trainer`` 进程的数据提供器。主要负责将用户的原始数据转换成 ``trainer`` 系统可以识别的数据类型。当系统需要新的数据训练时，会调用DataProvider获取数据接口。当所有数据读取完一轮后，DataProvider返回空数据通知系统一轮数据读取结束。 ``trainer`` 在每一轮训练开始时会重置DataProvider。

需要注意的是，DataProvider是被 ``trainer`` 系统调用，而不是新数据驱动系统；数据 ``shuffle`` 和一些随机化噪声添加都应该在DataProvider中完成。

为了用户能够灵活的处理数据，PaddlePaddle提供了处理数据的Python接口（称为 `PyDataProvider`_ ）。 在 ``PyDataProvider`` 中，系统C++模块接管了shuffle、处理batch、GPU和CPU通信、双缓冲、异步读取等问题，需要说明的是，一些情况下需要Python接口里处理shuffle，可以参考 `PyDataProvider`_ 的相关文档继续深入了解。


TrainerConfig
=============

模型配置是一个Python文件，主要包括神经网络结构、优化算法、数据传入方式，使用命令行参数 ``--config`` 传给``trainer``主程序。 例如\:

..	code-block:: bash

	paddle train --config=trainer_config.py

一个简单的模型配置文件为：

..  literalinclude:: trainer_config.py
    :linenos:

下面我们详细的介绍一下模型配置中各个模块的概念。


trainer_config_helpers
----------------------

PaddlePaddle配置文件与C++模块通信的最基础协议是 ``protobuf`` 。为了避免用户直接写比较难写的protobuf string，我们通过Python代码来生成protobuf包，这就是helpers的作用。所以在文件的开始，需要import这些helpers函数。

需要注意的是，这个 ``paddle.trainer_config_helpers`` 包是标准的python包，这意味着用户可以选择自己喜欢的 ``IDE`` 或者编辑器来编写Paddle的配置文件，这个Python包注释文档比较完善，并提供了IDE的代码提示与类型注释。

data_sources
------------

data_sources配置神经网络的数据源，使用的函数是 ``define_py_data_sources2`` ，这个函数是定义了使用 `PyDataProvider`_ 提供数据源。后缀 ``2`` 是Paddle历史遗留问题，因为Paddle之前使用的PyDataProvider性能问题，重构了一个新的 `PyDataProvider`_ 。

data_sources里通过train_list和test_list指定是训练文件列表和测试文件列表。 如果传入字符串的话，是指一个数据列表文件。这个数据列表文件中包含的是每一个训练或者测试文件的路径。如果传入一个list的话，则会默认生成一个list文件，再传入给train.list或者test.list。

其中``module`` 和``obj``指定了DataProvider的文件名和返回数据的函数名。更详细的使用，请参考 `PyDataProvider`_ 。

settings
--------

`settings`_ 设置训练神经网络所使用的算法。包括学习率、batch_size、优化算法、正则方法等，具体的使用方法请参考 `settings`_ 文档。

网络配置
--------

上述配置中余下的部分是神经网络配置，主要包括网络连接、 ``cost`` 层、评估器。

- 首先，定义了一个名字叫"pixel"的 ``data_layer`` ，每个layer返回的都是一个 ``LayerOutput`` 对象，比如第一层的输出对象称作 ``img`` 。
- 然后，这个对象作为另一个layer（ ``simple_img_conv_pool`` ）的输入， ``simple_img_conv_pool`` 是一个组合层，包括了图像的卷积 (convolution) 和池化(pooling)，
- 其次，连接到全连接层(``fc_layer``)，再连接到一个含Softmax激活的全连接层。
- 最终，连接到cost层（ ``classification_cost`` ）， ``classification_cost`` 默认使用多类交叉熵损失函数和分类错误率统计评估器。标记网络输出的函数为 ``outputs`` ，网络的输出是神经网络的优化目标，神经网络训练的时候，实际上就是要最小化这个输出。

用该模型配置进行预测时，网络的输出也是通过 ``outputs`` 标记。


Layer、Projection、Operator
===========================

PaddlePaddle的网络是基于Layer来配置的。所谓的Layer即是神经网络的某一层，一般是封装了许多复杂操作的操作集合。比如最简单的 ``fc_layer`` ，包括矩阵乘法、多输入的求和、加Bias操作、激活( ``activation`` )函数操作。

..	code-block:: python

	data = data_layer(name='data', size=200)
	out = fc_layer(input=data, size=200, act=TanhActivation())

对于更灵活配置需求，PaddlePaddle提供了基于 ``Projection`` 或者 ``Operator`` 的配置，这些需要与 ``mixed_layer`` 配合使用。 ``mixed_layer`` 是将多个输入累加求和，然后加Bias和 ``activation`` 操作。 ``mixed_layer`` 具体计算是通过内部的Projection和Operator完成。Projection含有可学习参数；而Operator不含可学习的参数，输入全是其他Layer的输出。


例如，和 ``fc_layer`` 同样功能的 ``mixed_layer`` 是:

..	code-block:: python

	data = data_layer(name='data', size=200)
	with mixed_layer(size=200) as out:
		out += full_matrix_projection(input=data)

PaddlePaddle可以使用 ``mixed layer`` 配置出非常复杂的网络，甚至可以直接配置一个完整的LSTM。用户可以参考 `mixed_layer`_ 的相关文档进行配置。

如何利用单机的所有GPU或所有CPU核心
===============================

PaddlePaddle的单机 ``trainer`` 进程可以充分利用一台计算机上所有的GPU资源或者CPU。

如果要使用机器上多块GPU，使用如下命令即可\:

..	code-block:: bash

	paddle train --use_gpu=true --trainer_count=4  # use 4 gpu card, 0, 1, 2, 3

如果要使用机器上多块CPU, 使用如下命令即可\:

..	code-block:: bash

	paddle train --trainer_count=4  # use 4 cpu cores.

如果要指定GPU编号，例如选择第0、2号GPU，则可以设置 ``CUDA_VISIBLE_DEVICES`` 环境变量来指定特定的GPU。具体可以参考连接`masking-gpu`_ ，命令为：

..	code-block:: bash

	env CUDA_VISIBLE_DEVICES=0,2 paddle train --use_gpu=true --trainer_count=2

如何利用多台机器的计算资源训练神经网络
===================================

PaddlePaddle多机采用经典的 ``Parameter Server`` 架构对多个节点的 ``trainer`` 进行同步。多机训练神经网络，要讲数据切分到不同的机器上，切分数据相对简单，所以在PaddlePaddle的开源实现中并没有提供相关工具包。

多机训练的经典拓扑结构如下\:

..	graphviz:: pserver_topology.dot

图中每个灰色方块是一台机器，在每个机器中，先启动一个 ``paddle pserver`` 进程，并指定端口号，可能的参数是\:

..	code-block:: bash

	paddle pserver --port=5000 --num_gradient_servers=4 --nics='eth0'

这里说明系统的 ``pserver`` 进程端口是 ``5000`` ，有四个训练进程(即 ``--gradient_servers=4`` ，PaddlePaddle同时将 ``trainer`` 称作 ``GradientServer`` 。因为其为负责提供Gradient)。 启动之后 ``pserver`` 进程之后，需要 ``trainer`` 训练进程，再在各个机器上运行如下命令\:

..	code-block:: bash

	paddle train --port=5000 --pservers=192.168.100.101,192.168.100.102,192.168.100.103,192.168.100.104 --config=...

对于简单的多机协同训练使用上述方式即可。另外，pserver/train 通常在高级情况下，还需要设置下面两个参数\：

* --ports_num\: 一个 pserver进程共绑定多少个端口用来做稠密更新。默认是1
* --ports_num_for_sparse\: 一个pserver进程共绑定多少端口用来做稀疏更新，默认是0

使用手工指定端口数量，是因为Paddle的网络通信中，使用了 ``int32`` 作为消息长度，比较容易在大模型下溢出。所以，在 ``pserver`` 进程中可以启动多个子线程去接受trainer的数据，这样单个子线程的长度就不会溢出了。但是这个值不可以调的过大，因为增加这个值，对性能尤其是内存占用有一定的开销，另外稀疏更新的端口如果太大的话，很容易导致某一个参数服务器没有分配到任何参数。

详细的说明可以参考，使用 `集群训练Paddle`_ 。


..  _PyDataProvider: ../ui/data_provider/pydataprovider2.html
..	_settings: ../../doc/ui/api/trainer_config_helpers/optimizers.html#settings
..	_mixed_layer: ../../doc/ui/api/trainer_config_helpers/layers.html#mixed-layer
..	_masking-gpu: http://www.acceleware.com/blog/cudavisibledevices-masking-gpus
..  _集群训练Paddle: ../cluster/index.html
