PaddlePaddle常见问题
====================

..  contents::


如何减少PaddlePaddle的内存占用
------------------------------

神经网络的训练本身是一个非常消耗内存和显存的工作。经常会消耗数十G的内存和数G的显存。
PaddlePaddle的内存占用主要分为如下几个方面:

* DataProvider缓冲池内存 (只针对内存)
* 神经元激活内存 （针对内存和显存）
* 参数内存 (针对内存和显存)
* 其他内存杂项

这其中，其他内存杂项是指PaddlePaddle本身所用的一些内存，包括字符串分配，临时变量等等，
这些内存就不考虑如何缩减了。

其他的内存的减少方法依次为


减少DataProvider缓冲池内存
++++++++++++++++++++++++++

PyDataProvider使用的是异步加载，同时在内存里直接随即选取数据来做Shuffle。即

..  graphviz::

    digraph {
        rankdir=LR;
        数据文件 -> 内存池 -> PaddlePaddle训练
    }

所以，减小这个内存池即可减小内存占用，同时也可以加速开始训练前数据载入的过程。但是，这
个内存池实际上决定了shuffle的粒度。所以，如果将这个内存池见小，又要保证数据是随机的，
那么最好将数据文件在每次读取之前做一次shuffle。可能的代码为

..  code-block::  python

    @provider(min_pool_size=0, ...)
    def process(settings, filename):
        os.system('shuf %s > %s.shuf' % (filename, filename))  # shuffle before.
        with open('%s.shuf' % filename, 'r') as f:
            for line in f:
                yield get_sample_from_line(line)

这样做可以极大的减少内存占用，并且可能会加速训练过程。 详细文档参考 `这里
<../ui/data_provider/pydataprovider2.html#provider>`_ 。

神经元激活内存
++++++++++++++

神经网络在训练的时候，会对每一个激活暂存一些数据，包括激活，參差等等。
在反向传递的时候，这些数据会被用来更新参数。这些数据使用的内存主要和两个参数有关系，
一是batch size，另一个是每条序列(Sequence)长度。所以，其实也是和每个mini-batch中包含
的时间步信息成正比。

所以，做法可以有两种。他们是

* 减小batch size。 即在网络配置中 `settings(batch_size=1000)` 设置成一个小一些的值。
* 减小序列的长度，或者直接扔掉非常长的序列。比如，一个数据集大部分序列长度是100-200,
  但是突然有一个10000长的序列，就很容易导致内存超限。特别是在LSTM等RNN中。

参数内存
++++++++

PaddlePaddle支持非常多的优化算法(Optimizer)，不同的优化算法需要使用不同大小的内存。
例如如果使用 :code:`adadelta` 算法，则需要使用参数规模大约5倍的内存。 如果参数保存下来的
文件为 :code:`100M`， 那么该优化算法至少需要 :code:`500M` 的内存。

可以考虑使用一些优化算法，例如 :code:`momentum`。

