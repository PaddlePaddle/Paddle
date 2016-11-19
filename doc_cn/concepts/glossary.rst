..	_glossary:

########################
Paddle文档中使用的词汇表
########################

..  _glossary_paddle:

PaddlePaddle
------------

TBD

.. _glossary_encode:

encode
------

参考\ :ref:`glossary_encoder`\ 。

..	_glossary_encoder:

encoder
-------

TBD

..  _glossary_sample:

样本
----

TBD Sample的概念

..  _glossary_lstm:

LSTM
----

TBD

..  _glossary_memory:

Memory
------

Memory是 :ref:`glossary_paddle` 实现 :ref:`glossary_RNN` 时候使用的一个概念。 :ref:`glossary_RNN` 即时间递归神经网络，通常要求时间步之间具有一些依赖性，即当前时间步下的神经网络依赖前一个时间步神经网络中某一个神经元输出。如下图所示。

..  graphviz:: glossary_rnn.dot

上图中虚线的连接，即是跨越时间步的网络连接。:ref:`glossary_paddle` 在实现 :ref:`glossary_RNN` 的时候，将这种跨越时间步的连接用一个特殊的神经网络单元实现。这个神经网络单元就叫Memory。Memory可以缓存上一个时刻某一个神经元的输出，然后在下一个时间步输入给另一个神经元。使用Memory的 :ref:`glossary_RNN` 实现便如下图所示。

..  graphviz:: glossary_rnn_with_memory.dot

使用这种方式，:ref:`glossary_paddle` 可以比较简单的判断哪些输出是应该跨越时间步的，哪些不是。

..	_glossary_timestep:

时间步
------

参考 :ref:`_glossary_Sequence` 。

..  _glossary_Sequence:

时间序列
--------

时间序列(time series)是指一系列的特征数据。这些特征数据之间的顺序是有意义的。即特征的数组，而不是特征的集合。而这每一个数组元素，或者每一个系列里的特征数据，即为一个时间步(time step)。值得注意的是，时间序列、时间步的概念，并不真正的和『时间』有关。只要一系列特征数据中的『顺序』是有意义的，即为时间序列的输入。

举例说明，例如文本分类中，我们通常将一句话理解成一个时间序列。比如一句话中的每一个单词，会变成词表中的位置。而这一句话就可以表示成这些位置的数组。例如 :code:`[9, 2, 3, 5, 3]` 。

关于时间序列(time series)的更详细准确的定义，可以参考 `维基百科页面 Time series <https://en.wikipedia.org/wiki/Time_series>`_ 或者 `维基百科中文页面 时间序列 <https://zh.wikipedia.org/wiki/%E6%99%82%E9%96%93%E5%BA%8F%E5%88%97>`_ 。

另外，Paddle中经常会将时间序列成为 :code:`Sequence` 。他们在Paddle的文档和API中是一个概念。 

..  _glossary_RNN:

RNN
---

RNN 在 :ref:`glossary_paddle` 的文档中，一般表示 :code:`Recurrent neural network`，即时间递归神经网络。详细介绍可以参考 `维基百科页面 Recurrent neural network <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_ 或者 `中文维基百科页面 <https://zh.wikipedia.org/wiki/%E9%80%92%E5%BD%92%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_ 中关于时间递归神经网络的介绍。

RNN 一般在 :ref:`glossary_paddle` 中，指对于一个 :ref:`glossary_Sequence` 输入数据，每一个时间步之间的神经网络具有一定的相关性。例如，某一个神经元的一个输入为上一个时间步网络中某一个神经元的输出。或者，从每一个时间步来看，神经网络的网络结构中具有有向环结构。

..  _glossary_双层RNN:

双层RNN
-------

双层RNN顾名思义，即 :ref:`glossary_RNN` 之间有一次嵌套关系。输入数据整体上是一个时间序列，而对于每一个内层特征数据而言，也是一个时间序列。即二维数组，或者数组的数组这个概念。 而双层RNN是可以处理这种输入数据的网络结构。

例如，对于段落的文本分类，即将一段话进行分类。我们将一段话看成句子的数组，每个句子又是单词的数组。这便是一种双层RNN的输入数据。而将这个段落的每一句话用lstm编码成一个向量，再对每一句话的编码向量用lstm编码成一个段落的向量。再对这个段落向量进行分类，即为这个双层RNN的网络结构。
