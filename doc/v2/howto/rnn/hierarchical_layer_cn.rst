###########################
支持双层序列作为输入的Layer
###########################

..	contents::

概述
====

在自然语言处理任务中，序列是一种常见的数据类型。一个独立的词语，可以看作是一个非序列输入，或者，我们称之为一个0层的序列；由词语构成的句子，是一个单层序列；若干个句子构成一个段落，是一个双层的序列。

双层序列是一个嵌套的序列，它的每一个元素，又是一个单层的序列。这是一种非常灵活的数据组织方式，帮助我们构造一些复杂的输入信息。

我们可以按照如下层次定义非序列，单层序列，以及双层序列。

+ 0层序列：一个独立的元素，类型可以是PaddlePaddle支持的任意输入数据类型
+ 单层序列：排成一列的多个元素，每个元素是一个0层序列，元素之间的顺序是重要的输入信息
+ 双层序列：排成一列的多个元素，每个元素是一个单层序列，称之为双层序列的一个子序列（subseq），subseq的每个元素是一个0层序列

在 PaddlePaddle中，下面这些Layer能够接受双层序列作为输入，完成相应的计算。

pooling
========

pooling 的使用示例如下。

..	code-block:: bash

        seq_pool = pooling(input=layer,
                           pooling_type=pooling.Max(),
                           agg_level=AggregateLevel.TO_SEQUENCE)
        
- `pooling_type` 目前支持两种，分别是：pooling.Max()和pooling.Avg()。

- `agg_level=AggregateLevel.TO_NO_SEQUENCE` 时（默认值）：

  - 作用：双层序列经过运算变成一个0层序列，或单层序列经过运算变成一个0层序列
  - 输入：一个双层序列，或一个单层序列
  - 输出：一个0层序列，即整个输入序列（单层或双层）的平均值（或最大值）

- `agg_level=AggregateLevel.TO_SEQUENCE` 时：

  - 作用：一个双层序列经过运算变成一个单层序列
  - 输入：必须是一个双层序列
  - 输出：一个单层序列，序列的每个元素是原来双层序列每个subseq元素的平均值（或最大值）

last_seq 和 first_seq
=====================

last_seq 的使用示例如下（first_seq 类似）。

..	code-block:: bash

        last = last_seq(input=layer,
                        agg_level=AggregateLevel.TO_SEQUENCE)
        
- `agg_level=AggregateLevel.TO_NO_SEQUENCE` 时（默认值）：

  - 作用：一个双层序列经过运算变成一个0层序列，或一个单层序列经过运算变成一个0层序列
  - 输入：一个双层序列或一个单层序列
  - 输出：一个0层序列，即整个输入序列（双层或者单层）最后一个，或第一个元素。

- `agg_level=AggregateLevel.TO_SEQUENCE` 时：
  - 作用：一个双层序列经过运算变成一个单层序列
  - 输入：必须是一个双层序列
  - 输出：一个单层序列，其中每个元素是双层序列中每个subseq最后一个（或第一个）元素。

expand
======

expand 的使用示例如下。

..	code-block:: bash

        ex = expand(input=layer1,
                    expand_as=layer2,
                    expand_level=ExpandLevel.FROM_NO_SEQUENCE)
        
- `expand_level=ExpandLevel.FROM_NO_SEQUENCE` 时（默认值）：

  - 作用：一个0层序列经过运算扩展成一个单层序列，或者一个双层序列
  - 输入：layer1必须是一个0层序列，是待扩展的数据；layer2 可以是一个单层序列，或者是一个双层序列，提供扩展的长度信息
  - 输出：一个单层序列或一个双层序列，输出序列的类型（双层序列或单层序列）和序列中含有元素的数目同 layer2 一致。若输出是单层序列，单层序列的每个元素（0层序列），都是对layer1元素的拷贝；若输出是双层序列，双层序列每个subseq中每个元素（0层序列），都是对layer1元素的拷贝

- `expand_level=ExpandLevel.FROM_SEQUENCE` 时：

  - 作用：一个单层序列经过运算扩展成一个双层序列
  - 输入：layer1必须是一个单层序列，是待扩展的数据；layer2 必须是一个双层序列，提供扩展的长度信息
  - 输出：一个双层序列，序列中含有元素的数目同 layer2 一致。要求单层序列含有元素的数目（0层序列）和双层序列含有subseq 的数目一致。单层序列第i个元素（0层序列），被扩展为一个单层序列，构成了输出双层序列的第i个 subseq 。
