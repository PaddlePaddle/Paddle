PyDataProvider2的使用
=====================

PyDataProvider是PaddlePaddle使用Python提供数据的推荐接口。使用该接口用户可以只关注如何
从文件中读取每一条数据，而不用关心数据如何传输给PaddlePaddle，数据如何存储等等。该数据
接口使用多线程读取数据，并提供了简单的Cache功能。


简单的使用场景
--------------

这里以MNIST手写识别为例，来说明简单的PyDataProvider如何使用。MNIST是一个包含有
70,000张灰度图片的数字分类数据集。对于MNIST而言，标签是0-9的数字，而特征即为
28*28的像素灰度值。这里我们使用简单的文本文件表示MNIST图片，样例数据如下。

..  literalinclude:: mnist_train.txt

其数据使用;间隔，第一段数据为这张图片的label，第二段数据为这个图片的像素值。
首先我们将这个数据文件(例如文件名是'mnist_train.txt')写入train.list。那么
train.list即为

..  literalinclude:: train.list

那么对应的dataprovider既为

..  literalinclude:: mnist_provider.py
    :linenos:

其中第一行是引入PaddlePaddle的PyDataProvider2包。主要函数是process函数。process函数
具有两个参数，第一个参数是 settings 。这个参数在这个样例里没有使用，具
体可以参考 settings 。第二个参数是filename，这个参数被PaddlePaddle进程传入，为
train.list中的一行(即train.list若干数据文件路径的某一个路径)。

:code:`@provider` 是一个Python的 `Decorator <http://www.learnpython.org/en/Decorators>`_
。这行的作用是设置DataProvider的一些属性，并且标记process函数是一个DataProvider。
如果不了解 `Decorator <http://www.learnpython.org/en/Decorators>`_ 是什么也没关系，
只需要知道这只是一个标记属性的方法就可以了。

属性 `input_types`_ 是设置这个DataProvider返回什么样的数据。这里设置的是返回一个
28*28的稠密向量和一个[0-9]，10维的整数值。 `input_types`_ 具体可以设置成什么其他格
式，请参考 `input_types`_ 的文档。

process函数是实现数据输入的主函数，在这个函数中，实现了打开文本文件，从文本文件中读取
每一行，并将每行转换成和 `input_types`_ 一致的特征，并在23行返回给PaddlePaddle进程。需要注意
的是， 返回的顺序需要和 `input_types`_ 中定义的顺序一致。

同时，返回数据在PaddlePaddle中是仅仅返回一条完整的训练样本，并且使用关键词 :code:`yield` 。
在PyDataProvider中，可以为一个数据文件返回多条训练样本(就像这个样例一样)，只需要在
process函数调用多次 :code:`yield` 即可。 :code:`yield` 是Python的一个关键词，相关的概
念是 :code:`generator` 。使用这个关键词，可以在一个函数里，多次返回变量。

在训练配置里，只需要使用一行代码即可以设置训练引用这个DataProvider。这个设置为

..  literalinclude:: mnist_config.py

这里说明了训练数据是 'train.list'，而没有测试数据。引用的DataProvider是 'mnist_provider' 
这个模块中的 'process' 函数。

同时，根据模型配置文件中 :code:`data_layer` 的名字，用户也可以显式指定返回的数据对应关系。例如:

.. literalinclude:: mnist_provider.dict.py
   :linenos:

如果用户不指定返回数据的对应关系，那么PaddlePaddle会粗略的根据layer的声明顺序，
来确定对应关系。这个对应关系可能不正确。所以推荐使用显式指定返回值和数据对应关系。

至此，简单的PyDataProvider样例就说明完毕了。对于用户来说，讲数据发送给PaddlePaddle，仅仅需要
知道如何从 **一个文件** 里面读取 **一条** 样本。而PaddlePaddle进程帮助用户做了

* 将数据组合成Batch训练
* Shuffle训练数据
* 多线程数据读取
* 缓存训练数据到内存(可选)
* CPU->GPU双缓存

是不是很简单呢？

序列模型数据提供
----------------

序列模型是指数据的某一维度是一个序列形式，即包含时间步信息。所谓时间步信息，
不一定和时间有关系，只是说明数据的顺序是重要的。例如，文本信息就是一个序列
数据。

这里举例的数据是英文情感分类的数据。数据是给一段英文文本，分类成正面情绪和
负面情绪两类(用0和1表示)。样例数据为

..  literalinclude:: sentimental_train.txt

这里，DataProvider可以是

..  literalinclude:: sentimental_provider.py

这个序列模型比较复杂。主要是增加了初始化机制。其中 :code:`on_init` 函数是使用
`@provider`_ 中的 `init_hook`_ 配置参数配置给DataProvider的。这个函数会在
DataProvider创建的时候执行。这个初始化函数具有如下参数:

* 第一个参数是 settings 对象。
* 其他参数均使用key word argument形式传入。有部分参数是Paddle自动生成的，
  参考 `init_hook`_ 。这里的 :code:`dictionary` 是从训练配置传入的dict对象。
  即从单词字符串到单词id的字典。

传入这个变量的方式为

..  literalinclude:: sentimental_config.py

这个声明基本上和mnist的样例一致。除了

* 在配置中读取了字典
* 在声明DataProvider的时候传入了dictionary作为参数。

在 :code:`on_init` 函数中，配置了 `input_types` 。这个和在 `@provider`_ 中配置
`input_types` 效果一致，但是在 `on_init` 中配置 `input_types` 是在运行时执行的，所以
可以根据不同的数据配置不同的输入类型。这里的输入特征是词id的序列，所以将 :code:`seq_type`
设置成了序列(同时，也可以使用 :code:`integer_sequence` 类型来设置)。

同时，将字典存入了settings 对象。这个字典可以在 :code:`process` 函数中使用。 :code:`process`
函数中的 settings 和 :code:`on_init` 中的settings 是同一个对象。

而在 :code:`process` 函数中，基本的处理逻辑也和mnist逻辑一致。依次返回了文件中的每条数据。

至此，基本的PyDataProvider使用介绍完毕了。具体DataProvider还具有什么功能，请参考下节reference。

参考(Reference)
---------------

@provider
+++++++++

:code:`@provider` 是一个Python的 `Decorator`_ ，他可以将某一个函数标记成一个PyDataProvider。它包含的参数有:

*  `input_types`_ 是数据输入格式。具体有哪些格式，参考 `input_types`_ 。
*  should_shuffle 是个DataProvider是不是要做shuffle，如果不设置的话，训练的时候默认shuffle，
   测试的时候默认不shuffle。
*  min_pool_size 是设置DataProvider在内存中最小暂存的数据条数。这个也是PaddlePaddle所能够保证的shuffle粒度。
   设置成-1的话，会预先读取全部数据到内存中。
*  pool_size 是设置DataProvider在内存中暂存的数据条数。设置成-1的话，即不在乎内存暂存多少条数据。
*  can_over_batch_size 表示是否允许Paddle暂存略微多余pool_size的数据。这样做可以避免很多死锁问题。
   一般推荐设置成True
*  calc_batch_size 传入的是一个函数，这个函数以一条数据为参数，返回batch_size的大小。默认情况下一条数据
   是一个batch size，但是有时为了计算均衡性，可以将一条数据设置成多个batch size
*  cache 是数据缓存的策略，参考 `cache`_
*  init_hook 是初始化时调用的函数，参考 `init_hook`_
*  check 设置成true的话，会根据input_types检查数据的合法性。
*  check_fail_continue 如果设置成true的话，即使在check中数据不合法，也会扔到这条数据，继续训练。 如果
   check是false的话，没有作用。

input_types
+++++++++++

PaddlePaddle的数据包括四种主要类型，和三种序列模式。其中，四种数据类型是

* dense_vector 表示稠密的浮点数向量。
* sparse_binary_vector 表示稀疏的零一向量，即大部分值为0，有值的位置只能取1
* sparse_float_vector 表示稀疏的向量，即大部分值为0，有值的部分可以是任何浮点数
* integer 表示整数标签。

而三种序列模式为

* SequenceType.NO_SEQUENCE 即不是一条序列
* SequenceType.SEQUENCE 即是一条时间序列
* SequenceType.SUB_SEQUENCE 即是一条时间序列，且序列的每一个元素还是一个时间序列。

不同的数据类型和序列模式返回的格式不同，列表如下

+----------------------+---------------------+-----------------------------------+------------------------------------------------+
|                      | NO_SEQUENCE         | SEQUENCE                          |  SUB_SEQUENCE                                  |
+======================+=====================+===================================+================================================+
| dense_vector         | [f, f, ...]         | [[f, ...], [f, ...], ...]         | [[[f, ...], ...], [[f, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_binary_vector | [i, i, ...]         | [[i, ...], [i, ...], ...]         | [[[i, ...], ...], [[i, ...], ...],...]         |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| sparse_float_vector  | [(i,f), (i,f), ...] | [[(i,f), ...], [(i,f), ...], ...] | [[[(i,f), ...], ...], [[(i,f), ...], ...],...] |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+
| integer_value        |  i                  | [i, i, ...]                       | [[i, ...], [i, ...], ...]                      |
+----------------------+---------------------+-----------------------------------+------------------------------------------------+

其中，f代表一个浮点数，i代表一个整数。

init_hook
+++++++++

init_hook可以传入一个函数。这个函数在初始化的时候会被调用。这个函数的参数是:

* 第一个参数是 settings 对象。这个对象和process的第一个参数一致。具有的属性有
    * settings.input_types 设置输入类型。参考 `input_types`_
    * settings.logger 一个logging对象
* 其他参数都使用key word argument传入。这些参数包括paddle定义的参数，和用户传入的参数。
    * Paddle定义的参数包括:
        * is_train bool参数，表示这个DataProvider是训练用的DataProvider或者测试用的
          DataProvider
        * file_list 所有文件列表。
    * 用户定义的参数使用args在训练配置中设置。

注意，PaddlePaddle保留添加参数的权力，所以init_hook尽量使用 :code:`**kwargs` , 来接受不使用的
函数来保证兼容性。

cache
+++++

DataProvider提供了两种简单的Cache策略。他们是

* CacheType.NO_CACHE 不缓存任何数据，每次都会从python端读取数据
* CacheType.CACHE_PASS_IN_MEM 第一个pass会从python端读取数据，剩下的pass会直接从内存里
  读取数据。 


注意事项
--------

可能的内存泄露问题
++++++++++++++++++

PaddlePaddle将train.list中的每一行，都传递给process函数，从而生成多个generator。
即如果train.list中，有100个训练文件，即会生成100个generator。这个本身不是一个很
严重的问题。

但是，如果在训练时，每一条训练数据都是一个文件，并且，训练数据非常多的情况下，就
会生成多个generator。每个generator在没有调用的时候，是几乎不占内存的。但是，当调
用过一次的时候，generator便会存下当前的上下文(Context)。而这个Context可能会非常
大。并且，generator至少调用两次才会知道是否停止。所以，即使在process里面只会有一
个yield，也需要两次随机选择到同样的generator的时候，才会释放该段内存。

..  code-block:: python

    def func():
        yield 0

    f = func()  # 创建generator
    tmp = next(f)  # 调用一次，返回0
    tmp = next(f)  # 调用第二次的时候，才会Stop Iteration

而如果按顺序调用这些generator就不会出现这个问题。

所以最佳实践推荐不要将每一个样本都放入train.list。而是将样本的地址放入另一个文本
文件，train.list写入那个文本文件的地址。 或者在python generator的上下文中尽量留
下非常少的变量引用。例如

..  code-block:: python

    def real_process(fn):
        # ... read from fn
        return result   # 当函数返回的时候，python可以解除掉内部变量的引用。

    def process(fn):
        yield real_process(fn)

这个问题是PyDataProvider读数据时候的逻辑问题，基本上不能整体修正。


内存不够用的情况
++++++++++++++++

PyDataProvider2会尽量使用内存。所以如果对于内存比较小的机器，推荐设置
:code:`pool_size` 变量，而这个变量推荐大于训练的batch size，并且在内存足够
的情况下越大越好。

