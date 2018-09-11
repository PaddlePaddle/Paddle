..  _user_guide_configure_simple_model:

##############
配置简单的网络
##############

在解决实际问题时，可以先从逻辑层面对问题进行建模，明确模型所需要的 **输入数据类型**、**计算逻辑**、**求解目标** 以及 **优化算法**。PaddlePaddle提供了丰富的算子来实现模型逻辑。下面以一个简单回归任务举例说明如何使用PaddlePaddle构建模型。该例子完整代码参见 `fit_a_line <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_fit_a_line.py>`_。

问题描述及定义
##############

问题描述: 给定一组数据 :math:`<X, Y>`，求解出函数 :math:`f`，使得 :math:`y=f(x)`，其中 :math:`x\subset X` 表示一条样本的特征，为 :math:`13` 维的实数向量；:math:`y \subset Y` 为一实数表示该样本对应的值。

我们可以尝试用回归模型来对问题建模，回归问题的损失函数有很多，这里选择常用的均方误差。为简化问题，这里假定 :math:`f` 为简单的线性变换函数，同时选用随机梯度下降算法来求解模型。

+----------------+----------------------------------------------+
| 输入数据类型   |  样本特征: 13 维 实数                        |
+                +----------------------------------------------+
|                |  样本标签: 1 维 实数                         |
+----------------+----------------------------------------------+
| 计算逻辑       | 使用线性模型，产生 1维实数作为模型的预测输出 |
+----------------+----------------------------------------------+
| 求解目标       | 最小化模型预测输出与样本标签间的均方误差     |
+----------------+----------------------------------------------+
| 优化算法       | 随机梯度下降                                 |
+----------------+----------------------------------------------+

使用PaddlePadle建模
###################

从逻辑层面明确了输入数据格式、模型结构、损失函数以及优化算法后，需要使用PaddlePaddle提供的API及算子来实现模型逻辑。一个典型的模型主要包含4个部分，分别是：输入数据格式定义，模型前向计算逻辑，损失函数以及优化算法。

数据层
------

PaddlePaddle提供了 :code:`fluid.layers.data()` 算子来描述输入数据的格式。

:code:`fluid.layers.data()` 算子的输出是一个Variable。这个Variable的实际类型是Tensor。Tensor具有强大的表征能力，可以表示多维数据。为了精确描述数据结构，通常需要指定数据shape以及数值类型type。其中shape为一个整数向量，type可以是一个字符串类型。目前支持的数据类型参考    :ref:`user_guide_paddle_support_data_types` 。 模型训练一般会使用batch的方式读取数据，而batch的size在训练过程中可能不固定。data算子会依据实际数据来推断batch size，所以这里提供shape时不用关心batch size，只需关心一条样本的shape即可，更高级用法请参考 :ref:`user_guide_customize_batch_size_rank`。从上知，:math:`x` 为 :math:`13` 维的实数向量，:math:`y` 为实数，可使用下面代码定义数据层：

.. code-block:: python

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

该模型使用的数据比较简单，事实上data算子还可以描述变长的、嵌套的序列数据。也可以使用 :code:`open_files` 打开文件进行训练。更详细的文档可参照 :ref:`user_guide_prepare_data`。

前向计算逻辑
------------

实现一个模型最重要的部分是实现计算逻辑，PaddlePaddle提供了丰富的算子。这些算子的封装粒度不同，通常对应一种或一组变换逻辑。算子输出即为对输入数据执行变换后的结果。用户可以灵活使用算子来完成复杂的模型逻辑。比如图像相关任务中会使用较多的卷积算子、序列任务中会使用LSTM/GRU等算子。复杂模型通常会组合多种算子，以完成复杂的变换。PaddlePaddle提供了非常自然的方式来组合算子，一般地可以使用下面的方式：

.. code-block:: python

    op_1_out = fluid.layers.op_1(input=op_1_in, ...)
    op_2_out = fluid.layers.op_2(input=op_1_out, ...)
    ...

其中op_1和op_2表示算子类型，可以是fc来执行线性变换(全连接)，也可以是conv来执行卷积变换等。通过算子的输入输出的连接来定义算子的计算顺序以及数据流方向。上面的例子中，op_1的输出是op_2的输入，那么在执行计算时，会先计算op_1，然后计算op_2。更复杂的模型可能需要使用控制流算子，依据输入数据来动态执行，针对这种情况，PaddlePaddle提供了IfElseOp和WhileOp等。算子的文档可参考 :code:`fluid.layers`。具体到这个任务, 我们使用一个fc算子：

.. code-block:: python

    y_predict = fluid.layers.fc(input=x, size=1, act=None)

损失函数
--------

损失函数对应求解目标，我们可以通过最小化损失来求解模型。大多数模型使用的损失函数，输出是一个实数值。但是PaddlePaddle提供的损失算子一般是针对一条样本计算。当输入一个batch的数据时，损失算子的输出有多个值，每个值对应一条样本的损失，所以通常会在损失算子后面使用mean等算子，来对损失做归约。模型在一次前向迭代后会得到一个损失值，PaddlePaddle会自动执行链式求导法则计算模型里面每个参数和变量对应的梯度值。这里使用均方误差损失：

.. code-block:: python

    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

优化方法
--------

确定损失函数后，可以通过前向计算得到损失值，然后通过链式求导法则得到参数的梯度值。获取梯度值后需要更新参数，最简单的算法是随机梯度下降法：:math:`w=w - \eta \cdot g`。但是普通的随机梯度下降算法存在一些问题: 比如收敛不稳定等。为了改善模型的训练速度以及效果，学术界先后提出了很多优化算法，包括： :code:`Momentum`、:code:`RMSProp`、:code:`Adam` 等。这些优化算法采用不同的策略来更新模型参数，一般可以针对具体任务和具体模型来选择优化算法。不管使用何种优化算法，学习率一般是一个需要指定的比较重要的超参数，需要通过实验仔细调整。这里采用随机梯度下降算法：

.. code-block:: python

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)

更多优化算子可以参考 :code:`fluid.optimizer()` 。

下一步做什么？
##############

使用PaddlePaddle实现模型时需要关注 **数据层**、**前向计算逻辑**、**损失函数** 和 **优化方法**。不同的任务需要的数据格式不同，涉及的计算逻辑不同，损失函数不同，优化方法也不同。PaddlePaddle提供了丰富的模型示例，可以以这些示例为参考来构建自己的模型结构。用户可以访问 `模型库 <https://github.com/PaddlePaddle/models/tree/develop/fluid>`_ 查看官方提供的示例。
