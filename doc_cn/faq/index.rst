PaddlePaddle常见问题
====================

..  contents::


如何减少PaddlePaddle的内存占用
------------------------------

神经网络的训练本身是一个非常消耗内存和显存的工作。经常会消耗数十G的内存和数G的显存。
PaddlePaddle的内存占用主要分为如下几个方面\:

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
个内存池实际上决定了shuffle的粒度。所以，如果将这个内存池减小，又要保证数据是随机的，
那么最好将数据文件在每次读取之前做一次shuffle。可能的代码为

..  literalinclude:: reduce_min_pool_size.py

这样做可以极大的减少内存占用，并且可能会加速训练过程。 详细文档参考 `这里
<../ui/data_provider/pydataprovider2.html#provider>`_ 。

神经元激活内存
++++++++++++++

神经网络在训练的时候，会对每一个激活暂存一些数据，包括激活，參差等等。
在反向传递的时候，这些数据会被用来更新参数。这些数据使用的内存主要和两个参数有关系，
一是batch size，另一个是每条序列(Sequence)长度。所以，其实也是和每个mini-batch中包含
的时间步信息成正比。

所以，做法可以有两种。他们是

* 减小batch size。 即在网络配置中 :code:`settings(batch_size=1000)` 设置成一个小一些的值。但是batch size本身是神经网络的超参数，减小batch size可能会对训练结果产生影响。
* 减小序列的长度，或者直接扔掉非常长的序列。比如，一个数据集大部分序列长度是100-200,
  但是突然有一个10000长的序列，就很容易导致内存超限。特别是在LSTM等RNN中。

参数内存
++++++++

PaddlePaddle支持非常多的优化算法(Optimizer)，不同的优化算法需要使用不同大小的内存。
例如如果使用 :code:`adadelta` 算法，则需要使用参数规模大约5倍的内存。 如果参数保存下来的
文件为 :code:`100M`， 那么该优化算法至少需要 :code:`500M` 的内存。

可以考虑使用一些优化算法，例如 :code:`momentum`。

如何加速PaddlePaddle的训练速度
------------------------------

PaddlePaddle是神经网络训练平台，加速PaddlePaddle训练有如下几个方面\：

* 减少数据载入的耗时
* 加速训练速度
* 利用更多的计算资源

减少数据载入的耗时
++++++++++++++++++

使用 :code:`pydataprovider`时，可以减少缓存池的大小，同时设置内存缓存功能，即可以极大的加速数据载入流程。
:code:`DataProvider` 缓存池的减小，和之前减小通过减小缓存池来减小内存占用的原理一致。

..  literalinclude:: reduce_min_pool_size.py

同时 :code:`@provider` 接口有一个 :code:`cache` 参数来控制缓存方法，将其设置成 :code:`CacheType.CACHE_PASS_IN_MEM` 的话，会将第一个 :code:`pass` (过完所有训练数据即为一个pass)生成的数据缓存在内存里，在之后的 :code:`pass` 中，不会再从 :code:`python` 端读取数据，而是直接从内存的缓存里读取数据。这也会极大减少数据读入的耗时。


加速训练速度
++++++++++++

PaddlePaddle支持Sparse的训练，sparse训练需要训练特征是 :code:`sparse_binary_vector` 、 :code:`sparse_vector` 、或者 :code:`integer_value` 的任一一种。同时，与这个训练数据交互的Layer，需要将其Parameter设置成 sparse 更新模式，即设置 :code:`sparse_update=True`

这里使用简单的 :code:`word2vec` 训练语言模型距离，具体使用方法为\:

使用一个词前两个词和后两个词，来预测这个中间的词。这个任务的DataProvider为\:

..  literalinclude:: word2vec_dataprovider.py

这个任务的配置为\:

..  literalinclude:: word2vec_config.py

更多关于sparse训练的内容请参考 `sparse训练的文档 <TBD>`_

利用更多的计算资源
++++++++++++++++++

利用更多的计算资源可以分为一下几个方式来进行\:

* 单机CPU训练
  * 使用多线程训练。设置命令行参数 :code:`trainer_count`，即可以设置参与训练的线程数量。使用方法为 :code:`paddle train --trainer_count=4`
* 单机GPU训练
  * 使用显卡训练。设置命令行参数 :code:`use_gpu`。 使用方法为 :code:`paddle train --use_gpu=true`
  * 使用多块显卡训练。设置命令行参数 :code:`use_gpu` 和 :code:`trainer_count`。使用 :code:`--use_gpu=True` 开启GPU训练，使用 :code:`trainer_count` 指定显卡数量。使用方法为 :code:`paddle train --use_gpu=true --trainer_count=4`
* 多机训练
  * 使用多机训练的方法也比较简单，需要先在每个节点启动 :code:`paddle pserver`，在使用 :code:`paddle train --pservers=192.168.100.1,192.168.100.2` 来指定每个pserver的ip地址
  * 具体的多机训练方法参考 `多机训练 <TBD>`_ 文档。


