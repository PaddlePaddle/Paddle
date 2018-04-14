..  _algo_hrnn_rnn_api_compare:

#####################
API comparision between RNN and hierarchical RNN
#####################

This article takes PaddlePaddle's hierarchical RNN unit test as an example. We will use multiple pairs of seperately uses of single-layer and hierarchical RNNs as the network configuration that have same effects to explain how to use hierarchical RNNs. All of the examples in this article only describe the API interface of the hierarchical RNN, while we do not use this hierarchical RNN to solve practical problems. If you want to understand the use of hierarchical RNN in specific issues, please refer to \ :ref:`algo_hrnn_demo`\ 。The unit test file used in this article's example is \ `test_RecurrentGradientMachine.cpp <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/test_RecurrentGradientMachine.cpp>`_\ 。

Example 1：Hierarchical RNN without Memory between subsequences
================================

The classical case in the hierarchical RNN is to perform sequence operations on each time series data in the inner layers seperately. And the sequence operations in the inner layers is independent, that is, it does not need to use Memory. 

In this example, the network configuration of single-layer RNNs and hierarchical RNNs are all to use LSTM as en encoder to compress a word-segmented sentence into a vector. The difference is that, RNN uses a hierarchical RNN model, treating multiple sentences as a whole to use encoder to compress simultaneously. They are completely consistent in their semantic meanings. This pair of semantically identical example configurations is as follows：

* RNN\: `sequence_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_layer_group.conf>`_
* Hierarchical RNN\: `sequence_nest_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_nest_layer_group.conf>`_


Reading hierarchical sequence data
----------------

Firstly, the original data in this example are as follows \:

- The original data in this example has 10 samples. Each of the sample includes two components: a lable(all 2 here), and a word-segmented sentence. This data is used by single RNN as well. 

..  literalinclude:: ../../../../paddle/gserver/tests/Sequence/tour_train_wdseg
    :language: text


- The data for hierarchical RNN has 4 samples. Every sample is seperated by a blank line, while the content of the data is the same as the original data. But as for hierarchical LSTM, the first sample will encode two sentences into two vectors simultaneously. The sentence count dealed simultaneously by this 4 samples are \ :code:`[2, 3, 2, 3]`\ .

..  literalinclude:: ../../../../paddle/gserver/tests/Sequence/tour_train_wdseg.nest
    :language: text

Secondly, as for these two types of different input data formats, the contrast of different DataProviders are as follows (`sequenceGen.py <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequenceGen.py>`_)\：

..  literalinclude:: ../../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 21-39
    :linenos:

- This is the DataProvider code for an ordinary single-layer time series. Its description is as follows: 
  
  * DataProvider returns two parts, that are "words" and "label"，as line 19 in the above code. 

    - "words" is a list of word table indices corresponding to each word in the sentence in the original data. Its data type is integer_value_sequence, that is integer list. So, "words" is a singler-layer time series in the data. 
    - "label" is the categorical label of each sentence, whose data type is integer_value. 

..  literalinclude:: ../../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 42-71
    :linenos:

- As for the same data, the DataProvider code for hierarchical time series. Its description is as follows: 

  - DataProvider returns two lists of data, that are "sentences" and "lables", corresponding to the sentences and lables in each group in the original data of hierarchical time series. 
  - "sentences" comes from the hierarchical time series original data. As it contains every sentences in each group internally, and each sentences are represented by a list of word table indices, so its data type is integer_value_sub_sequence, which is hierarchical time series. 
  - "labels" is the categorical lable of each sentence, so it is a sigle-layer time series. 


Model configuration
------------------------------------------

Firstly, let's look at the configuration of single-layer RNN. The hightlight part of line 9 to line 15 is the usage of single-layer RNN. Here we use the pre-defined RNN process function in PaddlePaddle. In this function, for each time step, RNN passes through an LSTM network. 

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_layer_group.conf
    :language: python
    :lines: 38-63
    :linenos:
    :emphasize-lines:  9-15


Secondly, let's look at the model configuration of hierarchical RNN which has the same semantic meaning. \:

* Many layers in PaddlePaddle does not care about whether the input is time series or not, e.g. \ :code:`embedding_layer`\ . In these layers, every operation is processed on each time step. 

* In the hightlight part of line 7 to line 26 of this configuration, we transforms the hierarchical time series data into single-layer time series data, then process each single-layer time series. 

  * Use the function \ :code:`recurrent_group`\ to transform. Input sequences need to be passed in when transforming. As we want to transform hierarchical time series into single-layer sequences, we need to lable the input data as \ :code:`SubsequenceInput`\ .
  
  * In this example, we disassemble every group of the original data into sentences using \ :code:`recurrent_group`\ . Each of the disassembled sentences passes through an LSTM network. This is equivalent to single-layer RNN configuration. 

* Similar to single-layer RNN configuration, we only use the last vector after the encode of LSTM. So we use the operation of \ :code:`last_seq`\ to \ :code:`recurrent_group`\ . But unlike single-layer RNN, we use the last element of every subsequence, so we need to set \ :code:`agg_level=AggregateLevel.TO_SEQUENCE`\ . 

* Till now, \ :code:`lstm_last`\ has the same result as \ :code:`lstm_last`\ in single-layer RNN configuration. 

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_nest_layer_group.conf
    :language: python
    :lines: 38-64
    :linenos:
    :emphasize-lines: 7-26

Example 2：Hierarchical RNN with Memory between subsequences
================================

This example is intended to implement two fully-equivalent full-connected RNNs using single-layer RNN and hierarchical RNN. 

* As for single-layer RNN, input is a full time series, e.g. \ :code:`[4, 5, 2, 0, 9, 8, 1, 4]`\ .

* As for hierarchical RNN, input is a hierarchial time series which elements are arbitrarily combination of data in single-layer RNN, e.g. \ :code:`[ [4, 5, 2], [0, 9], [8, 1, 4]]`. 

model configuration
------------------

We select the different parts between single-layer RNN and hierarchical RNN configurations, to compare and analyze the reason why they have same semantic meanings. 

- single-layer RNN：passes through a simple recurrent_group. For each time step, the current input y and the last time step's output rnn_state pass through a full-connected layer. 

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_rnn.conf
    :language: python
    :lines: 36-48

- hierarchical RNN, the outer layer's memory is an element. 

  - The recurrent_group of inner layer's inner_step is nearly the same as single-layer sequence, except for the case of boot_layer=outer_mem, which means using the outer layer's outer_mem as the initial state for the inner layer's memory. In the outer layer's out_step, outer_mem is the last vector of a subsequence, that is, the whole hierarchical group uses the last vector of the previous subsequence as the initial state for the next subsequence's memory. 
  - From the aspect of the input data, sentences from single-layer and hierarchical RNN are the same. The only difference is that, hierarchical RNN disassembes the sequence into subsequences. So in the hierarchical RNN configuration, we must use the last element of the previous subsequence as a boot_layer for the memory of the next subsequence, so that it makes no difference with "every time step uses the output of last time step" in the sigle-layer RNN configuration. 

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_nest_rnn.conf
    :language: python
    :lines: 39-66

..  warning::
    Currently PaddlePaddle only supports the case that the lengths of the time series of Memory in each time step are the same. 

Example 3：hierarchical RNN with unequal length inputs
==========================

.. role:: red

.. raw:: html

    <style> .red {color:red} </style>

**unequal length inputs** means in the multiple input sequences of recurrent_group, the lengths of subsequences can be unequal. But the output of the sequence, needs to be consistent with one of the input sequences. Using \ :red:`targetInlink`\ can help you specify which of the input sequences and the output sequence can be consistent, by default is the first input. 

The configurations of Example 3 are \ `sequence_rnn_multi_unequalength_inputs <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_rnn_multi_unequalength_inputs.py>`_ \ and \ `sequence_nest_rnn_multi_unequalength_inputs <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_nest_rnn_multi_unequalength_inputs.py>`_\ . 

The data for the configurations of Example 3's single-layer RNN and hierarchical RNN are exactly the same. 

* For the single-layer RNN, the data has two samples, which are \ :code:`[1, 2, 4, 5, 2], [5, 4, 1, 3, 1]`\ and \ :code:`[0, 2, 2, 5, 0, 1, 2], [1, 5, 4, 2, 3, 6, 1]`\ . Each of the data for the single-layer RNN has two group of features. 

* On the basis of the single-layer's data, hierarchical RNN's data randomly adds some partitions. For example, the first sample is transformed to \ :code:`[[0, 2], [2, 5], [0, 1, 2]],[[1, 5], [4], [2, 3, 6, 1]]`\ . 

* You need to pay attention that, PaddlePaddle only supports multiple input hierarchical RNNs that have same amount of subsequences currently. In this example, the two features both have 3 subsequences. Although the length of each subsequence can be different, the amount of subsequences should be the same. 


model configuration
--------

Similar to Example 2's configuration, Example 3's configuration uses single-layer and hierarchical RNN to implement 2 fully-equivalent full-connected RNNs. 

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

In the above code, the usage of single-layer and hierarchical RNNs are similar to Example 2, which difference is that it processes 2 inputs simultaneously. As for the hierarchical RNN, the lengths of the 2 input's subsequences are not equal. But we use the parameter \ :code:`targetInlink`\ to set the outper layer's \ :code:`recurrent_group`\ 's output format, so the shape of outer layer's output is the same as the shape of \ :code:`emb2`\ . 


Glossary
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
