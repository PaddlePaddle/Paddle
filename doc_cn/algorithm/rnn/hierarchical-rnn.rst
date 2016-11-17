#################
双层RNN配置与示例
#################

我们在 :code:`paddle/gserver/tests/test_RecurrentGradientMachine` 单测中，通过多组语义相同的单双层RNN配置，讲解如何使用双层RNN。

示例1：双进双出，subseq间无memory
=================================

配置：单层RNN（:code:`sequence_layer_group`）和双层RNN（:code:`sequence_nest_layer_group`），语义完全相同。

读取双层序列的方法
------------------

首先，我们看一下单双层序列的不同数据组织形式（您也可以采用别的组织形式）\:

- 单层序列的数据（ :code:`Sequence/tour_train_wdseg`）如下，一共有10个样本。每个样本由两部分组成，一个label（此处都为2）和一个已经分词后的句子。

..  literalinclude:: ../../../paddle/gserver/tests/Sequence/tour_train_wdseg
    :language: text


- 双层序列的数据（ :code:`Sequence/tour_train_wdseg.nest`）如下，一共有4个样本。样本间用空行分开，代表不同的双层序列，序列数据和上面的完全一样。每个样本的子句数分别为2,3,2,3。

..  literalinclude:: ../../../paddle/gserver/tests/Sequence/tour_train_wdseg.nest
    :language: text

其次，我们看一下单双层序列的不同dataprovider（见 :code:`sequenceGen.py` ）：

- 单层序列的dataprovider如下：
  
  - word_slot是integer_value_sequence类型，代表单层序列。
  - label是integer_value类型，代表一个向量。

..  literalinclude:: ../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 21-39

- 双层序列的dataprovider如下：

  - word_slot是integer_value_sub_sequence类型，代表双层序列。
  - label是integer_value_sequence类型，代表单层序列，即一个子句一个label。注意：也可以为integer_value类型，代表一个向量，即一个句子一个label。通常根据任务需求进行不同设置。
  - 关于dataprovider中input_types的详细用法，参见PyDataProvider2。

..  literalinclude:: ../../../paddle/gserver/tests/sequenceGen.py
    :language: python
    :lines: 42-71

模型中的配置
------------

首先，我们看一下单层序列的配置（见 :code:`sequence_layer_group.conf`）。注意：batchsize=5表示一次过5句单层序列，因此2个batch就可以完成1个pass。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_layer_group.conf
    :language: python
    :lines: 38-63


其次，我们看一下语义相同的双层序列配置（见 :code:`sequence_nest_layer_group.conf` ），并对其详细分析：

- batchsize=2表示一次过2句双层序列。但从上面的数据格式可知，2句双层序列和5句单层序列的数据完全一样。
- data_layer和embedding_layer不关心数据是否是序列格式，因此两个配置在这两层上的输出是一样的。
- lstmemory\:

  - 单层序列过了一个mixed_layer和lstmemory_group。
  - 双层序列在同样的mixed_layer和lstmemory_group外，直接加了一层group。由于这个外层group里面没有memory，表示subseq间不存在联系，即起到的作用仅仅是把双层seq拆成单层，因此双层序列过完lstmemory的输出和单层的一样。

- last_seq\:

  - 单层序列直接取了最后一个元素
  - 双层序列首先（last_seq层）取了每个subseq的最后一个元素，将其拼接成一个新的单层序列；接着（expand_layer层）将其扩展成一个新的双层序列，其中第i个subseq中的所有向量均为输入的单层序列中的第i个向量；最后（average_layer层）取了每个subseq的平均值。
  - 分析得出：第一个last_seq后，每个subseq的最后一个元素就等于单层序列的最后一个元素，而expand_layer和average_layer后，依然保持每个subseq最后一个元素的值不变（这两层仅是为了展示它们的用法，实际中并不需要）。因此单双层序列的输出是一样旳。

..  literalinclude:: ../../../paddle/gserver/tests/sequence_nest_layer_group.conf
    :language: python
    :lines: 38-84

示例2：双进双出，subseq间有memory
=================================

配置：单层RNN（ :code:`sequence_rnn.conf` ），双层RNN（ :code:`sequence_nest_rnn.conf` 和 :code:`sequence_nest_rnn_readonly_memory.conf` ），语义完全相同。

读取双层序列的方法
------------------

我们看一下单双层序列的不同数据组织形式和dataprovider（见 :code:`rnn_data_provider.py`）

..  literalinclude::  ../../../paddle/gserver/tests/rnn_data_provider.py
    :language: python
    :lines: 20-32

- 单层序列：有两句，分别为[1,3,2,4,5,2]和[0,2,2,5,0,1,2]。
- 双层序列：有两句，分别为[[1,3,2],[4,5,2]]（2个子句）和[[0,2],[2,5],[0,1,2]]（3个子句）。
- 单双层序列的label都分别是0和1

模型中的配置
------------

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
