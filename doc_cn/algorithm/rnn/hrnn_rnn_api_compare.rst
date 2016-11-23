..  _algo_hrnn_rnn_api_compare:

#####################
单双层RNN API对比介绍
#####################

这篇教程主要介绍了\ :ref:`glossary_双层RNN`\ 的API接口。本文中的以PaddlePaddle的\ :ref:`glossary_双层RNN`\ 单元测试为示例，用多对效果完全相同的、分别使用单、双层RNN作为网络配置的模型，来讲解如何使用\ :ref:`glossary_双层RNN`\ 。本文中所有的例子，都只是介绍\ :ref:`glossary_双层RNN`\ 的API接口，并不是使用\ :ref:`glossary_双层RNN`\ 解决实际的问题。如果想要了解\ :ref:`glossary_双层RNN`\ 在具体问题中的使用，请参考\ :ref:`algo_hrnn_demo`\ 。文章中示例所使用的单元测试文件是\ `test_RecurrentGradientMachine.cpp <https://github.com/reyoung/Paddle/blob/develop/paddle/gserver/tests/test_RecurrentGradientMachine.cpp>`_\ 。

示例1：双层RNN，子序列间无Memory
================================

在\ :ref:`glossary_双层RNN`\ 中的经典情况是将内层的每一个\ :ref:`glossary_sequence`\ 数据，分别进行序列操作。并且内层的序列操作之间是独立没有依赖的，即不需要使用\ :ref:`glossary_Memory`\ 的。

在本问题中，单层\ :ref:`glossary_RNN`\ 和\ :ref:`glossary_双层RNN`\ 的网络配置，都是将每一句分好词后的句子，使用LSTM作为encoder，压缩成一个向量。区别是\ :ref:`glossary_RNN`\ 使用两层序列模型，将多句话看成一个整体，同时使用encoder压缩，二者语意上完全一致。这组语意相同的示例配置如下

* 单层\ :ref:`glossary_RNN`\: `sequence_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_layer_group.conf>`_
* :ref:`glossary_双层RNN`\: `sequence_nest_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_nest_layer_group.conf>`_


读取双层序列数据
----------------

首先，本示例中使用的原始数据如下\:

- 本里中的原始数据一共有10个样本。每个样本由两部分组成，一个label（此处都为2）和一个已经分词后的句子。这个数据也被单层\ :ref:`glossary_RNN`\ 网络直接使用。

..  literalinclude:: ../../../paddle/gserver/tests/Sequence/tour_train_wdseg
    :language: text


- 双层序列数据一共有4个样本。 每个样本间用空行分开，整体数据和原始数据完全一样。而对于双层序列的LSTM来说，第一条数据同时encode两条数据成两个向量。这四条数据同时处理的句子为\ :code:`[2, 3, 2, 3]`\ 。

..  literalinclude:: ../../../paddle/gserver/tests/Sequence/tour_train_wdseg.nest
    :language: text

其次，对于两种不同的输入数据类型，不同\ :ref:`glossary_DataProvider`\ 对比如下(`sequenceGen.py <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequenceGen.py>`_)\：

..  literalinclude:: ../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 21-39
    :linenos:

- 这是普通的单层\ :ref:`glossary_sequence`\ 的\ :ref:`glossary_DataProvider`\ 代码，其说明如下：
  
  * :ref:`glossary_DataProvider`\ 共返回两个数据，分别是words和label。即上述代码中的第19行。
  - words是原始数据中的每一句话，所对应的词表index数组。它是integer_value_sequence类型的，即整数数组。words即为这个数据中的单层\ :ref:`glossary_sequence`\ 。
  - label是原始数据中对于每一句话的分类标签，它是integer_value类型的。

..  literalinclude:: ../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 42-71
    :linenos:

- 这是对于同样的数据，本示例中双层\ :ref:`glossary_sequence`\ 的\ :ref:`glossary_DataProvider`\ 代码，其说明如下：

  - :ref:`glossary_DataProvider`\ 共返回两组数据，分别是sentences和labels。即在双层序列的原始数据中，每一组内的所有句子和labels
  - sentences是双层\ :ref:`glossary_sequence`\ 的数据。他内部包括了每组数据中的所有句子，又使用句子中每一个单词的词表index表示每一个句子，故为双层\ :ref:`glossary_sequence`\ 。类型为 integer_value_sub_sequence 。
  - labels是每组内每一个句子的标签，故而是一个单层\ :ref:`glossary_sequence`\ 。


:ref:`glossary_trainer_config`\ 的模型配置
------------------------------------------

首先，我们看一下单层\ :ref:`glossary_RNN`\ 的配置。代码中9-15行即为单层RNN序列的使用代码。这里使用了PaddlePaddle预定义好的\ :ref:`glossary_RNN`\ 处理函数。在这个函数中，\ :ref:`glossary_RNN`\ 对于每一个\ :ref:`glossary_timestep`\ 通过了一个LSTM网络。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_layer_group.conf
    :language: python
    :lines: 38-63
    :linenos:
    :emphasize-lines:  9-15


其次，我们看一下语义相同的\ :ref:`glossary_双层RNN`\ 的网络配置。

* PaddlePaddle中的许多layer并不在意输入是否是\ :ref:`glossary_sequence`\ ，例如\ :code:`embedding_layer`\ 。在这些layer中，所有的操作都是针对每一个\ :ref:`glossary_timestep`\ 来进行的。

* 在该配置中，7-26行将双层\ :ref:`glossary_sequence`\ 数据，先变换成单层\ :ref:`glossary_sequence`\ 数据，在对每一个单层\ :ref:`glossary_sequence`\ 进行处理。

  * 使用\ :code:`recurrent_group`\ 这个函数进行变换，在变换时需要将输入序列传入。由于我们想要的变换是双层\ :ref:`glossary_sequence`\ => 单层\ :ref:`glossary_sequence`\ ，所以我们需要将输入数据标记成\ :code:`SubsequenceInput`\ 。
  
  * 在本例中，我们将原始数据的每一组，通过\ :code:`recurrent_group`\ 进行拆解，拆解成的每一句话再通过一个LSTM网络。这和单层\ :ref:`glossary_RNN`\ 的配置是等价的。

* 与单层\ :ref:`glossary_RNN`\ 的配置类似，我们只需要知道使用LSTM encode成的最后一个向量。所以对\ :code:`recurrent_group`\ 进行了\ :code:`last_seq`\ 操作。但是，和单层\ :ref:`glossary_RNN`\ 有区别的地方是，我们是对每一个子序列取最后一个元素。于是我们设置\ :code:`agg_level=AggregateLevel.EACH_SEQUENCE`\ 。

* 至此，\ :code:`lstm_last`\ 便和单层\ :ref:`glossary_RNN`\ 的配置中的\ :code:`lstm_last`\ 具有相同的结果了。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_nest_layer_group.conf
    :language: python
    :lines: 38-64
    :linenos:
    :emphasize-lines: 7-26

示例2：:ref:`glossary_双层RNN`，子序列间有\ :ref:`glossary_Memory`
==================================================================

本示例中，意图使用单层\ :ref:`glossary_RNN`\ 和\ :ref:`glossary_双层RNN`\ 同时实现一个完全等价的全连接\ :ref:`glossary_RNN`\ 。对于单层\ :ref:`glossary_RNN`\ ，输入数据为一个完整的\ :ref:`glossary_sequence`\ ，例如\ :code:`[4, 5, 2, 0, 9, 8, 1, 4]`\ 。而对于\ :ref:`glossary_双层RNN`\ ，输入数据为在单层\ :ref:`glossary_RNN`\ 数据里面，任意将一些数据组合成双层\ :ref:`glossary_sequence`\ ，例如\ :code:`[ [4, 5, 2], [0, 9], [8, 1, 4]]`。

:ref:`glossary_trainer_config`\ 的模型配置
------------------------------------------

我们选取单双层序列配置中的不同部分，来对比分析两者语义相同的原因。

- 单层序列：过了一个很简单的recurrent_group。每一个时间步，当前的输入y和上一个时间步的输出rnn_state做了一个全链接。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_rnn.conf
    :language: python
    :lines: 36-48

- 双层序列，外层memory是一个元素：

  - 内层inner_step的recurrent_group和单层序列的几乎一样。除了boot_layer=outer_mem，表示将外层的outer_mem作为内层memory的初始状态。外层outer_step中，outer_mem是一个子句的最后一个向量，即整个双层group是将前一个子句的最后一个向量，作为下一个子句memory的初始状态。
  - 从输入数据上看，单双层序列的句子是一样的，只是双层序列将其又做了子序列划分。因此双层序列的配置中，必须将前一个子句的最后一个元素，作为boot_layer传给下一个子句的memory，才能保证和单层序列的配置中“每一个时间步都用了上一个时间步的输出结果”一致。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_nest_rnn.conf
    :language: python
    :lines: 39-66

- 双层序列，外层memory是单层序列：

  - 由于外层每个时间步返回的是一个子句，这些子句的长度往往不等长。因此当外层有is_seq=True的memory时，内层是**无法直接使用**它的，即内层memory的boot_layer不能链接外层的这个memory。
  - 如果内层memory想**间接使用**这个外层memory，只能通过`pooling_layer`、`last_seq`或`first_seq`这三个layer将它先变成一个元素。但这种情况下，外层memory必须有boot_layer，否则在第0个时间步时，由于外层memory没有任何seq信息，因此上述三个layer的前向会报出“**Check failed: input.sequenceStartPositions**”的错误。

示例3：双进双出，输入不等长
===========================

.. role:: red

.. raw:: html

    <style> .red {color:red} </style>

**输入不等长** 是指recurrent_group的多个输入在各时刻的长度可以不相等, 但需要指定一个和输出长度一致的input，用 :red:`targetInlink` 表示。参考配置：单层RNN（:code:`sequence_rnn_multi_unequalength_inputs.conf`），双层RNN（:code:`sequence_nest_rnn_multi_unequalength_inputs.conf`）

读取双层序列的方法
------------------

我们看一下单双层序列的数据组织形式和dataprovider（见 :code:`rnn_data_provider.py` ）

..  literalinclude:: ../../../paddle/gserver/tests/rnn_data_provider.py
    :language: python
    :lines: 69-97

data2 中有两个样本，每个样本有两个特征, 记fea1, fea2。

- 单层序列：两个样本分别为[[1, 2, 4, 5, 2], [5, 4, 1, 3, 1]] 和 [[0, 2, 2, 5, 0, 1, 2], [1, 5, 4, 2, 3, 6, 1]]
- 双层序列：两个样本分别为

  - **样本1**\：[[[1, 2], [4, 5, 2]], [[5, 4, 1], [3, 1]]]。fea1和fea2都分别有2个子句，fea1=[[1, 2], [4, 5, 2]], fea2=[[5, 4, 1], [3, 1]]
  - **样本2**\：[[[0, 2], [2, 5], [0, 1, 2]],[[1, 5], [4], [2, 3, 6, 1]]]。fea1和fea2都分别有3个子句， fea1=[[0, 2], [2, 5], [0, 1, 2]], fea2=[[1, 5], [4], [2, 3, 6, 1]]。<br/>
  - **注意**\：每个样本中，各特征的子句数目需要相等。这里说的“双进双出，输入不等长”是指fea1在i时刻的输入的长度可以不等于fea2在i时刻的输入的长度。如对于第1个样本，时刻i=2, fea1[2]=[4, 5, 2]，fea2[2]=[3, 1]，3≠2。

- 单双层序列中，两个样本的label都分别是0和1

模型中的配置
------------

单层RNN（ :code:`sequence_rnn_multi_unequalength_inputs.conf`）和双层RNN（ :code:`v.conf`）两个模型配置达到的效果完全一样，区别只在于输入为单层还是双层序列，现在我们来看它们内部分别是如何实现的。

- 单层序列\：

  - 过了一个简单的recurrent_group。每一个时间步，当前的输入y和上一个时间步的输出rnn_state做了一个全连接，功能与示例2中`sequence_rnn.conf`的`step`函数完全相同。这里，两个输入x1,x2分别通过calrnn返回最后时刻的状态。结果得到的encoder1_rep和encoder2_rep分别是单层序列，最后取encoder1_rep的最后一个时刻和encoder2_rep的所有时刻分别相加得到context。
  - 注意到这里recurrent_group输入的每个样本中，fea1和fea2的长度都分别相等，这并非偶然，而是因为recurrent_group要求输入为单层序列时，所有输入的长度都必须相等。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_rnn_multi_unequalength_inputs.conf
    :language: python
    :lines: 41-58

- 双层序列\：

  - 双层RNN中，对输入的两个特征分别求时序上的连续全连接(`inner_step1`和`inner_step2`分别处理fea1和fea2)，其功能与示例2中`sequence_nest_rnn.conf`的`outer_step`函数完全相同。不同之处是，此时输入`[SubsequenceInput(emb1), SubsequenceInput(emb2)]`在各时刻并不等长。
  - 函数`outer_step`中可以分别处理这两个特征，但我们需要用<font color=red>targetInlink</font>指定recurrent_group的输出的格式（各子句长度）只能和其中一个保持一致，如这里选择了和emb2的长度一致。
  - 最后，依然是取encoder1_rep的最后一个时刻和encoder2_rep的所有时刻分别相加得到context。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_nest_rnn_multi_unequalength_inputs.conf
    :language: python
    :lines: 41-89

示例4：beam_search的生成
========================

TBD


词汇表
======

..  _glossary_memory:

Memory
------

Memory是PaddlePaddle实现 :ref:`glossary_RNN` 时候使用的一个概念。 :ref:`glossary_RNN` 即时间递归神经网络，通常要求时间步之间具有一些依赖性，即当前时间步下的神经网络依赖前一个时间步神经网络中某一个神经元输出。如下图所示。

..  graphviz:: glossary_rnn.dot

上图中虚线的连接，即是跨越时间步的网络连接。PaddlePaddle在实现 :ref:`glossary_RNN` 的时候，将这种跨越时间步的连接用一个特殊的神经网络单元实现。这个神经网络单元就叫Memory。Memory可以缓存上一个时刻某一个神经元的输出，然后在下一个时间步输入给另一个神经元。使用Memory的 :ref:`glossary_RNN` 实现便如下图所示。

..  graphviz:: glossary_rnn_with_memory.dot

使用这种方式，PaddlePaddle可以比较简单的判断哪些输出是应该跨越时间步的，哪些不是。

..  _glossary_timestep:

时间步
------

参考 :ref:`glossary_sequence` 。


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

RNN 一般在PaddlePaddle中，指对于一个 :ref:`glossary_sequence` 输入数据，每一个时间步之间的神经网络具有一定的相关性。例如，某一个神经元的一个输入为上一个时间步网络中某一个神经元的输出。或者，从每一个时间步来看，神经网络的网络结构中具有有向环结构。

..  _glossary_双层RNN:

双层RNN
-------

双层RNN顾名思义，即 :ref:`glossary_RNN` 之间有一次嵌套关系。输入数据整体上是一个时间序列，而对于每一个内层特征数据而言，也是一个时间序列。即二维数组，或者数组的数组这个概念。 而双层RNN是可以处理这种输入数据的网络结构。

例如，对于段落的文本分类，即将一段话进行分类。我们将一段话看成句子的数组，每个句子又是单词的数组。这便是一种双层RNN的输入数据。而将这个段落的每一句话用lstm编码成一个向量，再对每一句话的编码向量用lstm编码成一个段落的向量。再对这个段落向量进行分类，即为这个双层RNN的网络结构。

