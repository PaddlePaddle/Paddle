..  _api_pydataprovider2:

PyDataProvider2的使用
=====================

PyDataProvider2是PaddlePaddle使用Python提供数据的推荐接口。该接口使用多线程读取数据，并提供了简单的Cache功能；同时可以使用户只关注如何从文件中读取每一条数据，而不用关心数据如何传输，如何存储等等。

..  contents::

MNIST的使用场景
---------------

我们以MNIST手写识别为例，来说明PyDataProvider2的简单使用场景。

样例数据
++++++++

MNIST是一个包含有70,000张灰度图片的数字分类数据集。样例数据 ``mnist_train.txt`` 如下：

..  literalinclude:: src/mnist_train.txt

其中每行数据代表一张图片，行内使用 ``;`` 分成两部分。第一部分是图片的标签，为0-9中的一个数字；第二部分是28*28的图片像素灰度值。 对应的 ``train.list`` 即为这个数据文件的名字：

..  literalinclude:: src/train.list

dataprovider的使用
++++++++++++++++++

..  literalinclude:: src/mnist_provider.dict.py

- 首先，引入PaddlePaddle的PyDataProvider2包。
- 其次，定义一个Python的 `Decorator <http://www.learnpython.org/en/Decorators>`_ `@provider`_ 。用于将下一行的数据输入函数标记成一个PyDataProvider2，同时设置它的input_types属性。
  
  - `input_types`_：设置这个PyDataProvider2返回什么样的数据。本例根据网络配置中 ``data_layer`` 的名字，显式指定返回的是一个28*28维的稠密浮点数向量和一个[0-9]的10维整数标签。

    ..  literalinclude:: src/mnist_config.py
         :lines: 9-10

  - 注意：如果用户不显示指定返回数据的对应关系，那么PaddlePaddle会根据layer的声明顺序，来确定对应关系。但这个关系可能不正确，所以推荐使用显式指定的方式来设置input_types。
- 最后，实现数据输入函数（如本例的 ``process`` 函数）。

  - 该函数的功能是：打开文本文件，读取每一行，将行中的数据转换成与input_types一致的格式，然后返回给PaddlePaddle进程。注意，
    
    - 返回的顺序需要和input_types中定义的顺序一致。
    - 返回时，必须使用Python关键词 ``yield`` ，相关概念是 ``generator`` 。
    - 一次yield调用，返回一条完整的样本。如果想为一个数据文件返回多条样本，只需要在函数中调用多次yield即可（本例中使用for循环进行多次调用）。
  
  - 该函数具有两个参数：
  
    - settings：在本例中没有使用，具体可以参考 `init_hook`_ 中的说明。
    - filename：为 ``train.list`` 或 ``test.list`` 中的一行，即若干数据文件路径的某一个。

网络配置中的调用
++++++++++++++++

在网络配置里，只需要一行代码就可以调用这个PyDataProvider2，如，

..  literalinclude:: src/mnist_config.py
     :lines: 1-7

训练数据是 ``train.list`` ，没有测试数据，调用的PyDataProvider2是 ``mnist_provider`` 模块中的 ``process`` 函数。

小结
+++++

至此，简单的PyDataProvider2样例就说明完毕了。对用户来说，仅需要知道如何从 **一个文件** 中读取 **一条样本** ，就可以将数据传送给PaddlePaddle了。而PaddlePaddle则会帮用户做以下工作：

* 将数据组合成Batch进行训练
* 对训练数据进行Shuffle
* 多线程的数据读取
* 缓存训练数据到内存(可选)
* CPU->GPU双缓存

是不是很简单呢？

时序模型的使用场景
------------------
样例数据
++++++++

时序模型是指数据的某一维度是一个序列形式，即包含时间步信息。所谓时间步信息，不一定和时间有关系，只是说明数据的顺序是重要的。例如，文本信息就是一个序列数据。

本例采用英文情感分类的数据，即将一段英文文本数据，分类成正面情绪和负面情绪两类(用0和1表示)。样例数据 ``sentimental_train.txt`` 如下：

..  literalinclude:: src/sentimental_train.txt

dataprovider的使用
++++++++++++++++++

相对MNIST而言，这个dataprovider较复杂，主要原因是增加了初始化机制 `init_hook`_。本例的 ``on_init`` 函数就是根据该机制配置的，它会在dataprovider创建的时候执行。

- 其中 ``input_types`` 和在 `@provider`_ 中配置的效果一致。本例中的输入特征是词ID的序列，因此使用 ``integer_value_sequence`` 类型来设置。
- 将 ``dictionary`` 存入settings对象，在 ``process`` 函数中使用。 dictionary是从网络配置中传入的dict对象，即一个将单词字符串映射到单词ID的字典。

..  literalinclude:: src/sentimental_provider.py

网络配置中的调用
++++++++++++++++

调用这个PyDataProvider2的方法，基本上和MNIST样例一致，除了

* 在配置中需要读取外部字典。
* 在声明DataProvider的时候传入dictionary作为参数。

..  literalinclude:: src/sentimental_config.py
     :emphasize-lines: 12-14

参考(Reference)
---------------

@provider
+++++++++

``@provider`` 是一个Python的 `Decorator`_ ，可以将某一个函数标记成一个PyDataProvider2。如果不了解 `Decorator`_ 是什么也没关系，只需知道这是一个标记属性的方法就可以了。它包含的属性参数如下:

*  input_types：数据输入格式。具体的格式说明，请参考 `input_types`_ 。
*  should_shuffle：是不是要对数据做Shuffle。训练时默认shuffle，测试时默认不shuffle。
*  min_pool_size：设置内存中最小暂存的数据条数，也是PaddlePaddle所能够保证的shuffle粒度。如果为-1，则会预先读取全部数据到内存中。
*  pool_size： 设置内存中暂存的数据条数。如果为-1（默认），则不在乎内存暂存多少条数据。如果设置，则推荐大于训练时batch size的值，并且在内存足够的情况下越大越好。
*  can_over_batch_size：是否允许暂存略微多余pool_size的数据。由于这样做可以避免很多死锁问题，一般推荐设置成True。
*  calc_batch_size：可以传入一个函数，用于自定义每条数据的batch size（默认为1）。
*  cache： 数据缓存的策略，具体请参考 `cache`_ 。
*  init_hook：初始化时调用的函数，具体请参考 `init_hook`_ 。
*  check：如果为true，会根据input_types检查数据的合法性。
*  check_fail_continue：如果为true，那么当check出数据不合法时，会扔到这条数据，继续训练或预测。（对check=false的情况，没有作用）

input_types
+++++++++++

PaddlePaddle的数据包括四种主要类型，和三种序列模式。

四种数据类型：

* dense_vector：稠密的浮点数向量。
* sparse_binary_vector：稀疏的01向量，即大部分值为0，但有值的地方必须为1。
* sparse_float_vector：稀疏的向量，即大部分值为0，但有值的部分可以是任何浮点数。
* integer：整数标签。

三种序列模式：

* SequenceType.NO_SEQUENCE：不是一条序列
* SequenceType.SEQUENCE：是一条时间序列
* SequenceType.SUB_SEQUENCE： 是一条时间序列，且序列的每一个元素还是一个时间序列。

不同的数据类型和序列模式返回的格式不同，列表如下：

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

注意：对sparse_binary_vector和sparse_float_vector，PaddlePaddle存的是有值位置的索引。例如，

- 对一个5维非序列的稀疏01向量 ``[0, 1, 1, 0, 0]`` ，类型是sparse_binary_vector，返回的是 ``[1, 2]`` 。
- 对一个5维非序列的稀疏浮点向量 ``[0, 0.5, 0.7, 0, 0]`` ，类型是sparse_float_vector，返回的是 ``[(1, 0.5), (2, 0.7)]`` 。

init_hook
+++++++++

init_hook可以传入一个函数。该函数在初始化的时候会被调用，其参数如下:

* 第一个参数是settings对象，它和数据传入函数的第一个参数（如本例中 ``process`` 函数的 ``settings`` 参数）必须一致。该对象具有以下两个属性：
    * settings.input_types：数据输入格式，具体请参考 `input_types`_ 。
    * settings.logger：一个logging对象。
* 其他参数使用 ``kwargs`` （key word arguments）传入，包括以下两种：
    * PaddlePaddle定义的参数: 1）is_train：bool型参数，表示用于训练或预测；2）file_list：所有文件列表。
    * 用户定义的参数：使用args在网络配置中设置。

注意：PaddlePaddle保留添加参数的权力，因此init_hook尽量使用 ``**kwargs`` 来接受不使用的函数以保证兼容性。

cache
+++++

PyDataProvider2提供了两种简单的Cache策略：

* CacheType.NO_CACHE：不缓存任何数据，每次都会从python端读取数据
* CacheType.CACHE_PASS_IN_MEM：第一个pass会从python端读取数据，剩下的pass会直接从内存里
  读取数据。 


注意事项
--------

可能的内存泄露问题
++++++++++++++++++

PaddlePaddle将train.list中的每一行都传递给process函数，从而生成多个generator。当训练数据非常多时，就会生成非常多的generator。

虽然每个generator在没有调用的时候，是几乎不占内存的；但当调用过一次后，generator便会存下当前的上下文(Context)，而这个Context可能会非常大。并且，generator至少需要调用两次才会知道是否停止。所以，即使process函数里面只有一个yield，也需要两次随机选择到相同generator的时候，才会释放该段内存。

..  code-block:: python

    def func():
        yield 0

    f = func()  # 创建generator
    tmp = next(f)  # 调用一次，返回0
    tmp = next(f)  # 调用第二次的时候，才会Stop Iteration

由于顺序调用这些generator不会出现上述问题，因此有两种解决方案：

1. **最佳推荐**：将样本的地址放入另一个文本文件，train.list写入那个文本文件的地址。即不要将每一个样本都放入train.list。
2. 在generator的上下文中尽量留下非常少的变量引用，例如

..  code-block:: python

    def real_process(fn):
        # ... read from fn
        return result   # 当函数返回的时候，python可以解除掉内部变量的引用。

    def process(fn):
        yield real_process(fn)

注意：这个问题是PyDataProvider读数据时候的逻辑问题，很难整体修正。

内存不够用的情况
++++++++++++++++

PyDataProvider2会尽可能多的使用内存。因此，对于内存较小的机器，推荐使用 ``pool_size`` 变量来设置内存中暂存的数据条。具体请参考 `@provider`_ 中的说明。

