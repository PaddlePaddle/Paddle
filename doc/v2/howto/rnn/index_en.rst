RNN Models
==========
Recurrent neural networks(RNN) are an important tool to model sequential data. PaddlePaddle provides flexible interface for building complex recurrent neural network. We will demonstrate how to use PaddlePaddle to build RNN models in the following 4 parts.

In the first part, we will guide you how to configure recurrent neural network in PaddlePaddle from simple to complex. First, we will use a vanilla recurrent neural network as an example to show how to configure recurrent neural network architecture. Then We will use the sequence to sequence model as an example to demonstrate how you can configure complex recurrent neural network models gradually.

..  toctree::
  :maxdepth: 1

  rnn_config_en.rst

Recurrent Group is the key unit to build complex recurrent neural network models. The second part describes related concepts and Basic principles of Recurrent Group, and give a detailed description of Recurrent Group API interface. In addition, it also introduces Sequence-level RNN(hierarchical sequence as input) and the usage of Recurrent Group in it.

..  toctree::
  :maxdepth: 1
  
  recurrent_group_en.md
  
In the third part, two-level sequence is demonstrated briefly and then layers supporting two-level sequence as input are listed and described respectively.

..  toctree::
  :maxdepth: 1
  
  hierarchical_layer_en.rst

In the last part, the unit test of hierarchical RNN is presented as an example to explain how to use hierarchical RNN. We will use two-level sequence RNN and single-layer sequence RNN which have same effects with former as the network configuration seperately in unit test.

..  toctree::
  :maxdepth: 1
  
  hrnn_rnn_api_compare_en.rst

