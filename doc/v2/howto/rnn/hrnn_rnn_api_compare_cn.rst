..  _algo_hrnn_rnn_api_compare:

#####################
单双层RNN API对比介绍
#####################

本文以PaddlePaddle的双层RNN单元测试为示例，用多对效果完全相同的、分别使用单双层RNN作为网络配置的模型，来讲解如何使用双层RNN。本文中所有的例子，都只是介绍双层RNN的API接口，并不是使用双层RNN解决实际的问题。如果想要了解双层RNN在具体问题中的使用，请参考\ :ref:`algo_hrnn_demo`\ 。本文中示例所使用的单元测试文件是\ `test_RecurrentGradientMachine.cpp <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/test_RecurrentGradientMachine.cpp>`_\ 。

示例1：双层RNN，子序列间无Memory
================================

在双层RNN中的经典情况是将内层的每一个时间序列数据，分别进行序列操作；并且内层的序列操作之间独立无依赖，即不需要使用Memory\ 。

在本示例中，单层RNN和双层RNN的网络配置，都是将每一句分好词后的句子，使用LSTM作为encoder，压缩成一个向量。区别是RNN使用两层序列模型，将多句话看成一个整体同时使用encoder压缩。二者语意上完全一致。这组语义相同的示例配置如下：

* 单层RNN\: `sequence_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_layer_group.conf>`_
* 双层RNN\: `sequence_nest_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_nest_layer_group.conf>`_


读取双层序列数据
----------------

首先，本示例中使用的原始数据如下\:

- 本例中的原始数据一共有10个样本。每个样本由两部分组成，一个label（此处都为2）和一个已经分词后的句子。这个数据也被单层RNN网络直接使用。

..  literalinclude:: ../../../../paddle/gserver/tests/Sequence/tour_train_wdseg
    :language: text


- 双层序列数据一共有4个样本。 每个样本间用空行分开，整体数据和原始数据完全一样。但于双层序列的LSTM来说，第一个样本同时encode两条数据成两个向量。这四条数据同时处理的句子数量为\ :code:`[2, 3, 2, 3]`\ 。

..  literalinclude:: ../../../../paddle/gserver/tests/Sequence/tour_train_wdseg.nest
    :language: text

其次，对于两种不同的输入数据类型，不同DataProvider对比如下(`sequenceGen.py <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequenceGen.py>`_)\：

..  literalinclude:: ../../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 21-39
    :linenos:

- 这是普通的单层时间序列的DataProvider代码，其说明如下：
  
  * DataProvider共返回两个数据，分别是words和label。即上述代码中的第19行。

    - words是原始数据中的每一句话，所对应的词表index数组。它是integer_value_sequence类型的，即整数数组。words即为这个数据中的单层时间序列。
    - label是原始数据中对于每一句话的分类标签，它是integer_value类型的。

..  literalinclude:: ../../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 42-71
    :linenos:

- 对于同样的数据，双层时间序列的DataProvider的代码。其说明如下：

  - DataProvider共返回两组数据，分别是sentences和labels。即在双层序列的原始数据中，每一组内的所有句子和labels
  - sentences是双层时间序列的数据。由于它内部包含了每组数据中的所有句子，且每个句子表示为对应的词表索引数组，因此它是integer_value_sub_sequence 类型的，即双层时间序列。
  - labels是每组内每个句子的标签，故而是一个单层时间序列。


模型配置的模型配置
------------------------------------------

首先，我们看一下单层RNN的配置。代码中9-15行(高亮部分)即为单层RNN序列的使用代码。这里使用了PaddlePaddle预定义好的RNN处理函数。在这个函数中，RNN对于每一个时间步通过了一个LSTM网络。

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_layer_group.conf
    :language: python
    :lines: 38-63
    :linenos:
    :emphasize-lines:  9-15


其次，我们看一下语义相同的双层RNN的网络配置\:

* PaddlePaddle中的许多layer并不在意输入是否是时间序列，例如\ :code:`embedding_layer`\ 。在这些layer中，所有的操作都是针对每一个时间步来进行的。

* 在该配置的7-26行(高亮部分)，将双层时间序列数据先变换成单层时间序列数据，再对每一个单层时间序列进行处理。

  * 使用\ :code:`recurrent_group`\ 这个函数进行变换，在变换时需要将输入序列传入。由于我们想要的变换是双层时间序列=> 单层时间序列，所以我们需要将输入数据标记成\ :code:`SubsequenceInput`\ 。
  
  * 在本例中，我们将原始数据的每一组，通过\ :code:`recurrent_group`\ 进行拆解，拆解成的每一句话再通过一个LSTM网络。这和单层RNN的配置是等价的。

* 与单层RNN的配置类似，我们只需要使用LSTM encode成的最后一个向量。所以对\ :code:`recurrent_group`\ 进行了\ :code:`last_seq`\ 操作。但和单层RNN不同，我们是对每一个子序列取最后一个元素，因此\ :code:`agg_level=AggregateLevel.TO_SEQUENCE`\ 。

* 至此，\ :code:`lstm_last`\ 便和单层RNN配置中的\ :code:`lstm_last`\ 具有相同的结果了。

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_nest_layer_group.conf
    :language: python
    :lines: 38-64
    :linenos:
    :emphasize-lines: 7-26

示例2：双层RNN，子序列间有Memory
================================

本示例意图使用单层RNN和双层RNN实现两个完全等价的全连接RNN。

* 对于单层RNN，输入数据为一个完整的时间序列，例如\ :code:`[4, 5, 2, 0, 9, 8, 1, 4]`\ 。

* 对于双层RNN，输入数据为在单层RNN数据里面，任意将一些数据组合成双层时间序列，例如\ :code:`[ [4, 5, 2], [0, 9], [8, 1, 4]]`。

模型配置的模型配置
------------------

我们选取单双层序列配置中的不同部分，来对比分析两者语义相同的原因。

- 单层RNN：过了一个很简单的recurrent_group。每一个时间步，当前的输入y和上一个时间步的输出rnn_state做了一个全链接。

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_rnn.conf
    :language: python
    :lines: 36-48

- 双层RNN，外层memory是一个元素：

  - 内层inner_step的recurrent_group和单层序列的几乎一样。除了boot_layer=outer_mem，表示将外层的outer_mem作为内层memory的初始状态。外层outer_step中，outer_mem是一个子句的最后一个向量，即整个双层group是将前一个子句的最后一个向量，作为下一个子句memory的初始状态。
  - 从输入数据上看，单双层序列的句子是一样的，只是双层序列将其又做了子序列划分。因此双层序列的配置中，必须将前一个子句的最后一个元素，作为boot_layer传给下一个子句的memory，才能保证和单层序列的配置中“每个时间步都用了上一个时间步的输出结果”一致。

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_nest_rnn.conf
    :language: python
    :lines: 39-66

..  warning::
    PaddlePaddle目前只支持在每个时间步中，Memory的时间序列长度一致的情况。

示例3：双层RNN，输入不等长
==========================

.. role:: red

.. raw:: html

    <style> .red {color:red} </style>

**输入不等长** 是指recurrent_group的多个输入序列，在每个时间步的子序列长度可以不相等。但序列输出时，需要指定与某一个输入的序列信息是一致的。使用\ :red:`targetInlink`\ 可以指定哪一个输入和输出序列信息一致，默认指定第一个输入。 

示例3的配置分别为\ `单层不等长RNN <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_rnn_multi_unequalength_inputs.py>`_\ 和\ `双层不等长RNN <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_nest_rnn_multi_unequalength_inputs.py>`_\ 。

示例3对于单层RNN和双层RNN数据完全相同。

* 对于单层RNN的数据一共有两个样本，他们分别是\ :code:`[1, 2, 4, 5, 2], [5, 4, 1, 3, 1]`\ 和\ :code:`[0, 2, 2, 5, 0, 1, 2], [1, 5, 4, 2, 3, 6, 1]`\ 。对于每一个单层RNN的数据，均有两组特征。

* 在单层数据的基础上，双层RNN数据随意加了一些隔断，例如将第一条数据转化为\ :code:`[[0, 2], [2, 5], [0, 1, 2]],[[1, 5], [4], [2, 3, 6, 1]]`\ 。

* 需要注意的是PaddlePaddle目前只支持子序列数目一样的多输入双层RNN。例如本例中的两个特征，均有三个子序列。每个子序列长度可以不一致，但是子序列的数目必须一样。


模型配置
--------

和示例2中的配置类似，示例3的配置使用了单层RNN和双层RNN，实现两个完全等价的全连接RNN。

* 单层RNN\:

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_rnn_multi_unequalength_inputs.py
    :language: python
    :lines: 42-59
    :linenos:

* 双层RNN\ \:

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_nest_rnn_multi_unequalength_inputs.py
    :language: python
    :lines: 41-80
    :linenos:

在上面代码中，单层和双层序列的使用和示例2中的示例类似，区别是同时处理了两个输入。而对于双层序列，两个输入的子序列长度也并不相同。但是，我们使用了\ :code:`targetInlink`\ 参数设置了外层\ :code:`recurrent_group`\ 的输出格式。所以外层输出的序列形状，和\ :code:`emb2`\ 的序列形状一致。


词汇表
======

..  _glossary_memory:

Memory
------

Memory是PaddlePaddle实现RNN时候使用的一个概念。RNN即时间递归神经网络，通常要求时间步之间具有一些依赖性，即当前时间步下的神经网络依赖前一个时间步神经网络中某一个神经元输出。如下图所示。

..  graphviz:: src/glossary_rnn.dot

上图中虚线的连接，即是跨越时间步的网络连接。PaddlePaddle在实现RNN的时候，将这种跨越时间步的连接用一个特殊的神经网络单元实现。这个神经网络单元就叫Memory。Memory可以缓存上一个时刻某一个神经元的输出，然后在下一个时间步输入给另一个神经元。使用Memory的RNN实现便如下图所示。

..  graphviz:: src/glossary_rnn_with_memory.dot

使用这种方式，PaddlePaddle可以比较简单的判断哪些输出是应该跨越时间步的，哪些不是。

..  _glossary_timestep:

时间步
------

参考时间序列。


..  _glossary_sequence:

时间序列
--------

时间序列(time series)是指一系列的特征数据。这些特征数据之间的顺序是有意义的。即特征的数组，而不是特征的集合。而这每一个数组元素，或者每一个系列里的特征数据，即为一个时间步(time step)。值得注意的是，时间序列、时间步的概念，并不真正的和『时间』有关。只要一系列特征数据中的『顺序』是有意义的，即为时间序列的输入。

举例说明，例如文本分类中，我们通常将一句话理解成一个时间序列。比如一句话中的每一个单词，会变成词表中的位置。而这一句话就可以表示成这些位置的数组。例如 :code:`[9, 2, 3, 5, 3]` 。

关于时间序列(time series)的更详细准确的定义，可以参考 `维基百科页面 Time series <https://en.wikipedia.org/wiki/Time_series>`_ 或者 `维基百科中文页面 时间序列 <https://zh.wikipedia.org/wiki/%E6%99%82%E9%96%93%E5%BA%8F%E5%88%97>`_ 。

另外，Paddle中经常会将时间序列成为 :code:`Sequence` 。他们在Paddle的文档和API中是一个概念。 

..  _glossary_RNN:

RNN
---

RNN 在PaddlePaddle的文档中，一般表示 :code:`Recurrent neural network`，即时间递归神经网络。详细介绍可以参考 `维基百科页面 Recurrent neural network <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_ 或者 `中文维基百科页面 <https://zh.wikipedia.org/wiki/%E9%80%92%E5%BD%92%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_ 中关于时间递归神经网络的介绍。

RNN 一般在PaddlePaddle中，指对于一个时间序列输入数据，每一个时间步之间的神经网络具有一定的相关性。例如，某一个神经元的一个输入为上一个时间步网络中某一个神经元的输出。或者，从每一个时间步来看，神经网络的网络结构中具有有向环结构。

..  _glossary_双层RNN:

双层RNN
-------

双层RNN顾名思义，即RNN之间有一次嵌套关系。输入数据整体上是一个时间序列，而对于每一个内层特征数据而言，也是一个时间序列。即二维数组，或者数组的数组这个概念。 而双层RNN是可以处理这种输入数据的网络结构。

例如，对于段落的文本分类，即将一段话进行分类。我们将一段话看成句子的数组，每个句子又是单词的数组。这便是一种双层RNN的输入数据。而将这个段落的每一句话用lstm编码成一个向量，再对每一句话的编码向量用lstm编码成一个段落的向量。再对这个段落向量进行分类，即为这个双层RNN的网络结构。

