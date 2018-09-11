RNN模型
===========
循环神经网络（RNN）是对序列数据建模的重要工具。PaddlePaddle提供了灵活的接口以支持复杂循环神经网络的构建。
这里将分为以下四个部分详细介绍如何使用PaddlePaddle搭建循环神经网络。

第一部分由浅入深的展示了使用PaddlePaddle搭建循环神经网络的全貌：首先以简单的循环神经网络（vanilla RNN）为例，
说明如何封装配置循环神经网络组件；然后更进一步的通过序列到序列（sequence to sequence）模型，逐步讲解如何构建完整而复杂的循环神经网络模型。

..  toctree::
  :maxdepth: 1

  rnn_config_cn.rst

Recurrent Group是PaddlePaddle中实现复杂循环神经网络的关键，第二部分阐述了PaddlePaddle中Recurrent Group的相关概念和原理，
对Recurrent Group接口进行了详细说明。另外，对双层RNN（对应的输入为双层序列）及Recurrent Group在其中的使用进行了介绍。

..  toctree::
  :maxdepth: 1

  recurrent_group_cn.md

第三部分对双层序列进行了解释说明，列出了PaddlePaddle中支持双层序列作为输入的Layer，并对其使用进行了逐一介绍。

..  toctree::
  :maxdepth: 1

  hierarchical_layer_cn.rst

第四部分以PaddlePaddle的双层RNN单元测试中的网络配置为示例，辅以效果相同的单层RNN网络配置作为对比，讲解了多种情况下双层RNN的使用。

..  toctree::
  :maxdepth: 1

  hrnn_rnn_api_compare_cn.rst
