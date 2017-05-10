####################
FAQ
####################

..  contents::

1. 如何减少内存占用
---------------------------------

神经网络的训练本身是一个非常消耗内存和显存的工作，经常会消耗数10GB的内存和数GB的显存。
PaddlePaddle的内存占用主要分为如下几个方面\:

* DataProvider缓冲池内存（只针对内存）
* 神经元激活内存（针对内存和显存）
* 参数内存 （针对内存和显存）
* 其他内存杂项

其中，其他内存杂项是指PaddlePaddle本身所用的一些内存，包括字符串分配，临时变量等等，暂不考虑在内。

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

..  literalinclude:: src/reduce_min_pool_size.py

这样做可以极大的减少内存占用，并且可能会加速训练过程，详细文档参考 :ref:`api_pydataprovider2` 。

神经元激活内存
++++++++++++++

神经网络在训练的时候，会对每一个激活暂存一些数据，如神经元激活值等。
在反向传递的时候，这些数据会被用来更新参数。这些数据使用的内存主要和两个参数有关系，
一是batch size，另一个是每条序列(Sequence)长度。所以，其实也是和每个mini-batch中包含
的时间步信息成正比。

所以做法可以有两种：

* 减小batch size。 即在网络配置中 :code:`settings(batch_size=1000)` 设置成一个小一些的值。但是batch size本身是神经网络的超参数，减小batch size可能会对训练结果产生影响。
* 减小序列的长度，或者直接扔掉非常长的序列。比如，一个数据集大部分序列长度是100-200,
  但是突然有一个10000长的序列，就很容易导致内存超限，特别是在LSTM等RNN中。

参数内存
++++++++

PaddlePaddle支持非常多的优化算法(Optimizer)，不同的优化算法需要使用不同大小的内存。
例如使用 :code:`adadelta` 算法，则需要使用等于权重参数规模大约5倍的内存。举例，如果参数保存下来的模型目录
文件为 :code:`100M`， 那么该优化算法至少需要 :code:`500M` 的内存。

可以考虑使用一些优化算法，例如 :code:`momentum`。

2. 如何加速PaddlePaddle的训练速度
---------------------------------

加速PaddlePaddle训练可以考虑从以下几个方面\：

* 减少数据载入的耗时
* 加速训练速度
* 利用分布式训练驾驭更多的计算资源

减少数据载入的耗时
++++++++++++++++++

使用\ :code:`pydataprovider`\ 时，可以减少缓存池的大小，同时设置内存缓存功能，即可以极大的加速数据载入流程。
:code:`DataProvider` 缓存池的减小，和之前减小通过减小缓存池来减小内存占用的原理一致。

..  literalinclude:: src/reduce_min_pool_size.py

同时 :code:`@provider` 接口有一个 :code:`cache` 参数来控制缓存方法，将其设置成 :code:`CacheType.CACHE_PASS_IN_MEM` 的话，会将第一个 :code:`pass` (过完所有训练数据即为一个pass)生成的数据缓存在内存里，在之后的 :code:`pass` 中，不会再从 :code:`python` 端读取数据，而是直接从内存的缓存里读取数据。这也会极大减少数据读入的耗时。


加速训练速度
++++++++++++

PaddlePaddle支持Sparse的训练，sparse训练需要训练特征是 :code:`sparse_binary_vector` 、 :code:`sparse_vector` 、或者 :code:`integer_value` 的任一一种。同时，与这个训练数据交互的Layer，需要将其Parameter设置成 sparse 更新模式，即设置 :code:`sparse_update=True`

这里使用简单的 :code:`word2vec` 训练语言模型距离，具体使用方法为\:

使用一个词前两个词和后两个词，来预测这个中间的词。这个任务的DataProvider为\:

..  literalinclude:: src/word2vec_dataprovider.py

这个任务的配置为\:

..  literalinclude:: src/word2vec_config.py


利用更多的计算资源
++++++++++++++++++

利用更多的计算资源可以分为一下几个方式来进行\:

* 单机CPU训练

  * 使用多线程训练。设置命令行参数 :code:`trainer_count`。

* 单机GPU训练

  * 使用显卡训练。设置命令行参数 :code:`use_gpu`。
  * 使用多块显卡训练。设置命令行参数 :code:`use_gpu` 和 :code:`trainer_count` 。

* 多机训练

  * 请参考 :ref:`cluster_train` 。


3. 遇到“非法指令”或者是“illegal instruction”
--------------------------------------------

PaddlePaddle使用avx SIMD指令提高cpu执行效率，因此错误的使用二进制发行版可能会导致这种错误，请选择正确的版本。

4. 如何选择SGD算法的学习率
--------------------------

在采用sgd/async_sgd进行训练时，一个重要的问题是选择正确的learning_rate。如果learning_rate太大，那么训练有可能不收敛，如果learning_rate太小，那么收敛可能很慢，导致训练时间过长。

通常做法是从一个比较大的learning_rate开始试，如果不收敛，那减少学习率10倍继续试验，直到训练收敛为止。那么如何判断训练不收敛呢？可以估计出如果模型采用不变的输出最小的cost0是多少。

如果训练过程的的cost明显高于这个常数输出的cost，那么我们可以判断为训练不收敛。举一个例子，假如我们是三分类问题，采用multi-class-cross-entropy作为cost，数据中0,1,2三类的比例为 :code:`0.2, 0.5, 0.3` , 那么常数输出所能达到的最小cost是 :code:`-(0.2*log(0.2)+0.5*log(0.5)+0.3*log(0.3))=1.03` 。如果训练一个pass（或者更早）后，cost还大于这个数，那么可以认为训练不收敛，应该降低学习率。


5. 如何初始化参数
-----------------

默认情况下，PaddlePaddle使用均值0，标准差为 :math:`\frac{1}{\sqrt{d}}` 来初始化参数。其中 :math:`d` 为参数矩阵的宽度。这种初始化方式在一般情况下不会产生很差的结果。如果用户想要自定义初始化方式，PaddlePaddle目前提供两种参数初始化的方式\:

* 高斯分布。将 :code:`param_attr` 设置成 :code:`param_attr=ParamAttr(initial_mean=0.0, initial_std=1.0)`
* 均匀分布。将 :code:`param_attr` 设置成 :code:`param_attr=ParamAttr(initial_max=1.0, initial_min=-1.0)`

比如设置一个全连接层的参数初始化方式和bias初始化方式，可以使用如下代码。

..  code-block:: python

    hidden = fc_layer(input=ipt, param_attr=ParamAttr(initial_max=1.0, initial_min=-1.0),
                      bias_attr=ParamAttr(initial_mean=1.0, initial_std=0.0))

上述代码将bias全部初始化为1.0, 同时将参数初始化为 :code:`[1.0, -1.0]` 的均匀分布。

6. 如何共享参数
---------------

PaddlePaddle的参数使用名字 :code:`name` 作为参数的ID，相同名字的参数，会共享参数。设置参数的名字，可以使用 :code:`ParamAttr(name="YOUR_PARAM_NAME")` 来设置。更方便的设置方式，是使得要共享的参数使用同样的 :code:`ParamAttr` 对象。

简单的全连接网络，参数共享的配置示例为\:

..  literalinclude:: ../../python/paddle/trainer_config_helpers/tests/configs/shared_fc.py

这里 :code:`hidden_a` 和 :code:`hidden_b` 使用了同样的parameter和bias。并且softmax层的两个输入也使用了同样的参数 :code:`softmax_param`。

7. \*-cp27mu-linux_x86_64.whl is not a supported wheel on this platform.
------------------------------------------------------------------------

出现这个问题的主要原因是，系统编译wheel包的时候，使用的 :code:`wheel` 包是最新的，
而系统中的 :code:`pip` 包比较老。具体的解决方法是，更新 :code:`pip` 包并重新编译PaddlePaddle。
更新 :code:`pip` 包的方法是\:

..  code-block:: bash

    pip install --upgrade pip

8.  python相关的单元测试都过不了
--------------------------------

如果出现以下python相关的单元测试都过不了的情况：

..  code-block:: bash

    24 - test_PyDataProvider (Failed)
    26 - test_RecurrentGradientMachine (Failed)
    27 - test_NetworkCompare (Failed)
    28 - test_PyDataProvider2 (Failed)
    32 - test_Prediction (Failed)
    33 - test_Compare (Failed)
    34 - test_Trainer (Failed)
    35 - test_TrainerOnePass (Failed)
    36 - test_CompareTwoNets (Failed)
    37 - test_CompareTwoOpts (Failed)
    38 - test_CompareSparse (Failed)
    39 - test_recurrent_machine_generation (Failed)
    40 - test_PyDataProviderWrapper (Failed)
    41 - test_config_parser (Failed)
    42 - test_swig_api (Failed)
    43 - layers_test (Failed)

并且查询PaddlePaddle单元测试的日志，提示：

..  code-block:: bash

    paddle package is already in your PYTHONPATH. But unittest need a clean environment.
    Please uninstall paddle package before start unittest. Try to 'pip uninstall paddle'.

解决办法是：

* 卸载PaddlePaddle包 :code:`pip uninstall paddle`, 清理掉老旧的PaddlePaddle安装包，使得单元测试有一个干净的环境。如果PaddlePaddle包已经在python的site-packages里面，单元测试会引用site-packages里面的python包，而不是源码目录里 :code:`/python` 目录下的python包。同时，即便设置 :code:`PYTHONPATH` 到 :code:`/python` 也没用，因为python的搜索路径是优先已经安装的python包。


9. 运行Docker GPU镜像出现 "CUDA driver version is insufficient"
----------------------------------------------------------------

用户在使用PaddlePaddle GPU的Docker镜像的时候，常常出现 `Cuda Error: CUDA driver version is insufficient for CUDA runtime version`, 原因在于没有把机器上CUDA相关的驱动和库映射到容器内部。
具体的解决方法是：

..  code-block:: bash

    $ export CUDA_SO="$(\ls usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run ${CUDA_SO} ${DEVICES} -it paddledev/paddlepaddle:latest-gpu

更多关于Docker的安装与使用, 请参考 `PaddlePaddle Docker 文档 <http://www.paddlepaddle.org/doc_cn/build_and_install/install/docker_install.html>`_ 。


10. CMake源码编译, 找到的PythonLibs和PythonInterp版本不一致
----------------------------------------------------------------

这是目前CMake寻找Python的逻辑存在缺陷，如果系统安装了多个Python版本，CMake找到的Python库和Python解释器版本可能有不一致现象，导致编译PaddlePaddle失败。正确的解决方法是，
用户强制指定特定的Python版本，具体操作如下：

    ..  code-block:: bash

        cmake .. -DPYTHON_EXECUTABLE=<exc_path> -DPYTHON_LIBRARY=<lib_path>  -DPYTHON_INCLUDE_DIR=<inc_path>

用户需要指定本机上Python的路径：``<exc_path>``, ``<lib_path>``, ``<inc_path>``

10. A protocol message was rejected because it was too big
----------------------------------------------------------

如果在训练NLP相关模型时，出现以下错误：

..  code-block:: bash

    [libprotobuf ERROR google/protobuf/io/coded_stream.cc:171] A protocol message was rejected because it was too big (more than 67108864 bytes).  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
    F1205 14:59:50.295174 14703 TrainerConfigHelper.cpp:59] Check failed: m->conf.ParseFromString(configProtoStr)

可能的原因是：传给dataprovider的某一个args过大，一般是由于直接传递大字典导致的。错误的define_py_data_sources2类似：

..  code-block:: python

     src_dict = dict()
     for line_count, line in enumerate(open(src_dict_path, "r")):
        src_dict[line.strip()] = line_count

     define_py_data_sources2(
        train_list,
        test_list,
        module="dataprovider",
        obj="process",
        args={"src_dict": src_dict})

解决方案是：将字典的地址作为args传给dataprovider，然后在dataprovider里面根据该地址加载字典。即define_py_data_sources2应改为：

..  code-block:: python

     define_py_data_sources2(
        train_list,
        test_list,
        module="dataprovider",
        obj="process",
        args={"src_dict_path": src_dict_path})

完整源码可参考 `seqToseq <https://github.com/PaddlePaddle/Paddle/tree/develop/demo/seqToseq>`_ 示例。

11. 如何指定GPU设备
-------------------

例如机器上有4块GPU，编号从0开始，指定使用2、3号GPU：

* 方式1：通过 `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ 环境变量来指定特定的GPU。

..      code-block:: bash

        env CUDA_VISIBLE_DEVICES=2,3 paddle train --use_gpu=true --trainer_count=2

* 方式2：通过命令行参数 ``--gpu_id`` 指定。

..      code-block:: bash

        paddle train --use_gpu=true --trainer_count=2 --gpu_id=2


12. 训练过程中出现 :code:`Floating point exception`, 训练因此退出怎么办?
------------------------------------------------------------------------

Paddle二进制在运行时捕获了浮点数异常，只要出现浮点数异常(即训练过程中出现NaN或者Inf)，立刻退出。浮点异常通常的原因是浮点数溢出、除零等问题。
主要原因包括两个方面:

* 训练过程中参数或者训练过程中的梯度尺度过大，导致参数累加，乘除等时候，导致了浮点数溢出。
* 模型一直不收敛，发散到了一个数值特别大的地方。
* 训练数据有问题，导致参数收敛到了一些奇异的情况。或者输入数据尺度过大，有些特征的取值达到数百万，这时进行矩阵乘法运算就可能导致浮点数溢出。

主要的解决办法是减小学习律或者对数据进行归一化处理。
