..  _algo_hrnn_rnn_api_compare:

#####################
API comparision between RNN and hierarchical RNN
#####################

This article takes PaddlePaddle's hierarchical RNN unit test as an example. We will use several examples to illestrate the usage of single-layer and hierarchical RNNs. Each example has two model configurations, one for single-layer, and the other for hierarchical RNN. Although the implementations are different, both the two model configurations' effects are the same. All of the examples in this article only describe the API interface of the hierarchical RNN, while we do not use this hierarchical RNN to solve practical problems. If you want to understand the use of hierarchical RNN in specific issues, please refer to \ :ref:`algo_hrnn_demo`\ 。The unit test file used in this article's example is \ `test_RecurrentGradientMachine.cpp <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/test_RecurrentGradientMachine.cpp>`_\ 。

Example 1：Hierarchical RNN without Memory between subsequences
================================

The classical case in the hierarchical RNN is to perform sequence operations on each time series data in the inner layers seperately. And the sequence operations in the inner layers is independent, that is, it does not need to use Memory. 

In this example, the network configuration of single-layer RNNs and hierarchical RNNs are all to use LSTM as en encoder to compress a word-segmented sentence into a vector. The difference is that, RNN uses a hierarchical RNN model, treating multiple sentences as a whole to use encoder to compress simultaneously. They are completely consistent in their semantic meanings. This pair of semantically identical example configurations is as follows：

* RNN\: `sequence_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_layer_group.conf>`_
* Hierarchical RNN\: `sequence_nest_layer_group.conf <https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/gserver/tests/sequence_nest_layer_group.conf>`_


Reading hierarchical sequence data
----------------

Firstly, the original data in this example is as follows \:

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

  - DataProvider returns two lists of data, that are "sentences" and "labels", corresponding to the sentences and labels in each group in the original data of hierarchical time series. 
  - "sentences" comes from the hierarchical time series original data. As it contains every sentences in each group internally, and each sentences are represented by a list of word table indices, so its data type is integer_value_sub_sequence, which is hierarchical time series. 
  - "labels" is the categorical lable of each sentence, so it is a sigle-layer time series. 


Model configuration
------------------------------------------

Firstly, let's look at the configuration of single-layer RNN. The hightlighted part of line 9 to line 15 is the usage of single-layer RNN. Here we use the pre-defined RNN process function in PaddlePaddle. In this function, for each time step, RNN passes through an LSTM network. 

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_layer_group.conf
    :language: python
    :lines: 38-63
    :linenos:
    :emphasize-lines:  9-15


Secondly, let's look at the model configuration of hierarchical RNN which has the same semantic meaning. \:

* Most layers in PaddlePaddle do not care about whether the input is time series or not, e.g. \ :code:`embedding_layer`\ . In these layers, every operation is processed on each time step. 

* In the hightlighted part of line 7 to line 26 of this configuration, we transform the hierarchical time series data into single-layer time series data, then process each single-layer time series. 

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

This example is intended to implement two fully-equivalent fully-connected RNNs using single-layer RNN and hierarchical RNN. 

* As for single-layer RNN, input is a full time series, e.g. \ :code:`[4, 5, 2, 0, 9, 8, 1, 4]`\ .

* As for hierarchical RNN, input is a hierarchical time series which elements are arbitrarily combination of data in single-layer RNN, e.g. \ :code:`[ [4, 5, 2], [0, 9], [8, 1, 4]]`. 

model configuration
------------------

We select the different parts between single-layer RNN and hierarchical RNN configurations, to compare and analyze the reason why they have same semantic meanings. 

- single-layer RNN：passes through a simple recurrent_group. For each time step, the current input y and the last time step's output rnn_state pass through a fully-connected layer. 

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

Similar to Example 2's configuration, Example 3's configuration uses single-layer and hierarchical RNN to implement 2 fully-equivalent fully-connected RNNs. 

* single-layer RNN\:

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_rnn_multi_unequalength_inputs.py
    :language: python
    :lines: 42-59
    :linenos:

* hierarchical RNN\ \:

..  literalinclude:: ../../../../paddle/gserver/tests/sequence_nest_rnn_multi_unequalength_inputs.py
    :language: python
    :lines: 41-80
    :linenos:

In the above code, the usage of single-layer and hierarchical RNNs are similar to Example 2, which difference is that it processes 2 inputs simultaneously. As for the hierarchical RNN, the lengths of the 2 input's subsequences are not equal. But we use the parameter \ :code:`targetInlink` \ to set the outper layer's \ :code:`recurrent_group` \ 's output format, so the shape of outer layer's output is the same as the shape of \ :code:`emb2`\ . 


Glossary
======

..  _glossary_memory:

Memory
------

Memory is a concept when PaddlePaddle is implementing RNN. RNN, recurrent neural network, usually requires some dependency between time steps, that is, the neural network in current time step depends on one of the neurons in the neural network in previous time steps, as the following figure shows: 

..  graphviz:: src/glossary_rnn.dot

The dotted connections in the figure, is the network connections across time steps. When PaddlePaddle is implementing RNN, this connection accross time steps is implemented using a special neural network unit, called Memory. Memory can cache the output of one of the neurons in previous time step, then can be passed to another neuron in next time step. The implementation of an RNN using Memory is as follows: 

..  graphviz:: src/glossary_rnn_with_memory.dot

With this method, PaddlePaddle can easily determine which outputs should cross time steps, and which should not. 

..  _glossary_timestep:

time step
------

refers to time series


..  _glossary_sequence:

time series
--------

Time series is a series of featured data. The order among these featured data is meaningful. So it is a list of features, not a set of features. As for each element of this list, or the featured data in each series, is called a time step. It must be noted that, the concepts of time series and time steps, are not necessarrily related to "time". As long as the "order" in a series of featured data is meaningful, it can be the input of time series. 

For example, in text classification task, we regard a sentence as a time series. So, each word in the sentence can become the index of the word in the word table. So this sentence can be represented as a list of these indices, e.g.:code:`[9, 2, 3, 5, 3]` . 

For a more detailed and accurate definition of the time series, please refer to `Wikipedia of Time series <https://en.wikipedia.org/wiki/Time_series>`_  or `Chinese Wikipedia of time series <https://zh.wikipedia.org/wiki/%E6%99%82%E9%96%93%E5%BA%8F%E5%88%97>`_  . 

In additioin, Paddle always calls time series as :code:`Sequence` . They are a same concept in Paddle's documentations and APIs. 

..  _glossary_RNN:

RNN
---

In PaddlePaddle's documentations, RNN is usually represented as :code:`Recurrent neural network` . For more information, please refer to `Wikipedia Recurrent neural network <https://en.wikipedia.org/wiki/Recurrent_neural_network>`_ or `Chinese Wikipedia <https://zh.wikipedia.org/wiki/%E9%80%92%E5%BD%92%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_ . 

In PaddlePaddle, RNN usually means, for the input data of a time series, the neural network between each time steps has a certain relevance. For example, the input of a certain neuron is the output of a certain neuron in the neural network of the last time step. Or, as for each time step, the network structure of the neural network has a directed ring structure. 

..  _glossary_hierarchical_RNN:

hierarchical RNN
-------

Hierarchical RNN, as the name suggests, means there is a nested relationship in RNNs. The input data is a time series, but for each of the inner featured data, it is also a time series, namely 2-dimentional array, or, array of array. Hierarchical RNN is a neural network that can process this type of input data. 

For example, the task of text classification of a paragragh, meaning to classify a paragraph of sentences. We can treat a paragraph as an array of sentences, and each sentence is an array of words. This is a type of the input data for the hierarchical RNN. We encode each sentence of this paragraph into a vector using LSTM, then encode each of the encoded vectors into a vector of this paragraph using LSTM. Finally we use this paragraph vector perform classification, which is the neural network structure of this hierarchical RNN. 

