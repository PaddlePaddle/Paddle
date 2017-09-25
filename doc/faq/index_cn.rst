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

7. paddlepaddle\*.whl is not a supported wheel on this platform.
------------------------------------------------------------------------

出现这个问题的主要原因是，没有找到和当前系统匹配的paddlepaddle安装包。最新的paddlepaddle python安装包支持Linux x86_64和MacOS 10.12操作系统，并安装了python 2.7和pip 9.0.1。

更新 :code:`pip` 包的方法是\:

..  code-block:: bash

    pip install --upgrade pip

如果还不行，可以执行 :code:`python -c "import pip; print(pip.pep425tags.get_supported())"` 获取当前系统支持的python包的后缀，
并对比是否和正在安装的后缀一致。

如果系统支持的是 :code:`linux_x86_64` 而安装包是 :code:`manylinux1_x86_64` ，需要升级pip版本到最新；
如果系统支持 :code:`manylinux1_x86_64` 而安装包（本地）是 :code:`linux_x86_64` ，可以重命名这个whl包为 :code:`manylinux1_x86_64` 再安装。

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

11. CMake源码编译，Paddle版本号为0.0.0
--------------------------------------

如果运行 :code:`paddle version`, 出现 :code:`PaddlePaddle 0.0.0`；或者运行 :code:`cmake ..`，出现

..  code-block:: bash

    CMake Warning at cmake/version.cmake:20 (message):
      Cannot add paddle version from git tag

那么用户需要拉取所有的远程分支到本机，命令为 :code:`git fetch upstream`，然后重新cmake即可。

12. A protocol message was rejected because it was too big
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

13. 如何指定GPU设备
-------------------

例如机器上有4块GPU，编号从0开始，指定使用2、3号GPU：

* 方式1：通过 `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ 环境变量来指定特定的GPU。

..      code-block:: bash

        env CUDA_VISIBLE_DEVICES=2,3 paddle train --use_gpu=true --trainer_count=2

* 方式2：通过命令行参数 ``--gpu_id`` 指定。

..      code-block:: bash

        paddle train --use_gpu=true --trainer_count=2 --gpu_id=2


14. 训练过程中出现 :code:`Floating point exception`, 训练因此退出怎么办?
------------------------------------------------------------------------

Paddle二进制在运行时捕获了浮点数异常，只要出现浮点数异常(即训练过程中出现NaN或者Inf)，立刻退出。浮点异常通常的原因是浮点数溢出、除零等问题。
主要原因包括两个方面:

* 训练过程中参数或者训练过程中的梯度尺度过大，导致参数累加，乘除等时候，导致了浮点数溢出。
* 模型一直不收敛，发散到了一个数值特别大的地方。
* 训练数据有问题，导致参数收敛到了一些奇异的情况。或者输入数据尺度过大，有些特征的取值达到数百万，这时进行矩阵乘法运算就可能导致浮点数溢出。

这里有两种有效的解决方法：

* 对梯度的值进行限制，可以通过设置 :code:`optimizer` 中的 :code:`gradient_clipping_threshold` 来预防梯度爆炸，具体可以参考  `nmt_without_attention  <https://github.com/PaddlePaddle/models/tree/develop/nmt_without_attention>`_ 示例。

* 由于最终的损失函数关于每一层输出对应的梯度都会遵循链式法则进行反向传播，因此，可以通过对每一层要传输的梯度大小进行限制来预防浮点数溢出。具体可以对特定的网络层的属性进行设置：:code:`layer_attr=paddle.attr.ExtraAttr(error_clipping_threshold=10.0)` 。完整代码可以参考示例 `machine translation <https://github.com/PaddlePaddle/book/tree/develop/08.machine_translation>`_ 。

除此之外，还可以通过减小学习律或者对数据进行归一化处理来解决这类问题。

15. 编译安装后执行 import paddle.v2 as paddle 报ImportError: No module named v2
------------------------------------------------------------------------
先查看一下是否曾经安装过paddle v1版本，有的话需要先卸载：

pip uninstall py_paddle paddle

然后安装paddle的python环境, 在build目录下执行

pip install python/dist/paddle*.whl && pip install ../paddle/dist/py_paddle*.whl

16. PaddlePaddle存储的参数格式是什么，如何和明文进行相互转化
---------------------------------------------------------

PaddlePaddle保存的模型参数文件内容由16字节头信息和网络参数两部分组成。头信息中，1~4字节表示PaddlePaddle版本信息，请直接填充0；5~8字节表示每个参数占用的字节数，当保存的网络参数为float类型时为4，double类型时为8；9~16字节表示保存的参数总个数。

将PaddlePaddle保存的模型参数还原回明文时，可以使用相应数据类型的 :code:`numpy.array` 加载具体网络参数，此时可以跳过PaddlePaddle模型参数文件的头信息。若在PaddlePaddle编译时，未指定按照double精度编译，默认情况下按照float精度计算，保存的参数也是float类型。这时在使用 :code:`numpy.array` 时，一般设置 :code:`dtype=float32` 。示例如下：

..  code-block:: python

    def read_parameter(fname, width):
        s = open(fname).read()
        # skip header
        vec = np.fromstring(s[16:], dtype=np.float32)
        # width is the size of the corresponding layer
        np.savetxt(fname + ".csv", vec.reshape(width, -1),
                fmt="%.6f", delimiter=",")


将明文参数转化为PaddlePaddle可加载的模型参数时，首先构造头信息，再写入网络参数。下面的代码将随机生成的矩阵转化为可以被PaddlePaddle加载的模型参数。

..  code-block:: python

    def gen_rand_param(param_file, width, height, need_trans):
        np.random.seed()
        header = struct.pack("iil", 0, 4, height * width)
        param = np.float32(np.random.rand(height, width))
        with open(param_file, "w") as fparam:
            fparam.write(header + param.tostring())

17. 如何加载预训练参数
------------------------------

* 对加载预训练参数的层，设置其参数属性 :code:`is_static=True`，使该层的参数在训练过程中保持不变。以embedding层为例，代码如下：

..  code-block:: python

    emb_para = paddle.attr.Param(name='emb', is_static=True)
    paddle.layer.embedding(size=word_dim, input=x, param_attr=emb_para)


* 从模型文件将预训练参数载入 :code:`numpy.array`，在创建parameters后，使用 :code:`parameters.set()` 加载预训练参数。PaddlePaddle保存的模型参数文件前16字节为头信息，用户将参数载入 :code:`numpy.array` 时须从第17字节开始。以embedding层为例，代码如下：

..  code-block:: python

    def load_parameter(file_name, h, w):
        with open(file_name, 'rb') as f:
            f.read(16)  # skip header.
            return np.fromfile(f, dtype=np.float32).reshape(h, w)

    parameters = paddle.parameters.create(my_cost)
    parameters.set('emb', load_parameter(emb_param_file, 30000, 256))

18. 集群多节点训练，日志中保存均为网络通信类错误
------------------------------

集群多节点训练，日志报错为网络通信类错误，比如 :code:`Connection reset by peer` 等。
此类报错通常是由于某一个节点的错误导致这个节点的训练进程退出，从而引发其他节点无法连接导致，可以参考下面的步骤排查：

* 从 :code:`train.log` ， :code:`server.log` 找到最早报错的地方，查看是否是其他错误引发的报错（比如FPE，内存不足，磁盘空间不足等）。

* 如果发现最早的报错就是网络通信的问题，很有可能是非独占方式执行导致的端口冲突，可以联系OP，看当前MPI集群是否支持resource=full参数提交，如果支持增加此参数提交，并更换job 端口。

* 如果当前MPI集群并不支持任务独占模式，可以联系OP是否可以更换集群或升级当前集群。

19. PaddlePaddle如何输出多个层
------------------------------

* 将需要输出的层作为 :code:`paddle.inference.Inference()` 接口的 :code:`output_layer` 参数输入，代码如下：

..  code-block:: python

    inferer = paddle.inference.Inference(output_layer=[layer1, layer2], parameters=parameters)

* 指定要输出的字段进行输出。以输出 :code:`value` 字段为例，代码如下：

..  code-block:: python

    out = inferer.infer(input=data_batch, flatten_result=False, field=["value"])

这里设置 :code:`flatten_result=False`，得到的输出结果是元素个数等于输出字段数的 :code:`list`，该 :code:`list` 的每个元素是由所有输出层相应字段结果组成的 :code:`list`，每个字段结果的类型是 :code:`numpy.array`。:code:`flatten_result` 的默认值为 :code:`True`，该情况下，PaddlePaddle会分别对每个字段将所有输出层的结果按行进行拼接，如果各输出层该字段 :code:`numpy.array` 结果的相应维数不匹配，程序将不能正常运行。

20. :code:`paddle.layer.memory` 的参数 :code:`name` 如何使用
-------------------------------------------------------------

* :code:`paddle.layer.memory` 用于获取特定layer上一时间步的输出，该layer是通过参数 :code:`name` 指定，即，:code:`paddle.layer.memory` 会关联参数 :code:`name` 取值相同的layer，并将该layer上一时间步的输出作为自身当前时间步的输出。

* PaddlePaddle的所有layer都有唯一的name，用户通过参数 :code:`name` 设定，当用户没有显式设定时，PaddlePaddle会自动设定。而 :code:`paddle.layer.memory` 不是真正的layer，其name由参数 :code:`memory_name` 设定，当用户没有显式设定时，PaddlePaddle会自动设定。:code:`paddle.layer.memory` 的参数 :code:`name` 用于指定其要关联的layer，需要用户显式设定。

21. dropout 使用
-----------------

* 在PaddlePaddle中使用dropout有两种方式

  * 在相应layer的 :code:`layer_atter` 设置 :code:`drop_rate`，以 :code:`paddle.layer.fc` 为例，代码如下：

  ..  code-block:: python

      fc = paddle.layer.fc(input=input, layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=0.5))

  * 使用 :code:`paddle.layer.dropout`，以 :code:`paddle.layer.fc` 为例，代码如下：

  ..  code-block:: python

      fc = paddle.layer.fc(input=input)
      drop_fc = paddle.layer.dropout(input=fc, dropout_rate=0.5)

* :code:`paddle.layer.dropout` 实际上使用了 :code:`paddle.layer.add_to`，并在该layer里采用第一种方式设置 :code:`drop_rate` 来使用dropout的。这种方式对内存消耗较大。

* PaddlePaddle在激活函数里实现dropout，而不是在layer里实现。

* :code:`paddle.layer.lstmemory`、:code:`paddle.layer.grumemory`、:code:`paddle.layer.recurrent` 不是通过一般的方式来实现对输出的激活，所以不能采用第一种方式在这几个layer里设置 :code:`drop_rate` 来使用dropout。若要对这几个layer使用dropout，可采用第二种方式，即使用 :code:`paddle.layer.dropout`。

22. 如何设置学习率退火（learning rate annealing）
------------------------------------------------

在相应的优化算法里设置learning_rate_schedule及相关参数，以使用Adam算法为例，代码如下：

..  code-block:: python

    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        learning_rate_decay_a=0.5,
        learning_rate_decay_b=0.75,
        learning_rate_schedule="poly",)

PaddlePaddle目前支持8种learning_rate_schedule，这8种learning_rate_schedule及其对应学习率计算方式如下：

* "constant"

  lr = learning_rate

* "poly"

  lr = learning_rate * pow(1 + learning_rate_decay_a * num_samples_processed, -learning_rate_decay_b)

  其中，num_samples_processed为已训练样本数，下同。

* "caffe_poly"

  lr = learning_rate * pow(1.0 - num_samples_processed / learning_rate_decay_a, learning_rate_decay_b)

* "exp"

  lr = learning_rate * pow(learning_rate_decay_a, num_samples_processed / learning_rate_decay_b)

* "discexp"

  lr = learning_rate * pow(learning_rate_decay_a, floor(num_samples_processed / learning_rate_decay_b))

* "linear"

  lr = max(learning_rate - learning_rate_decay_a * num_samples_processed, learning_rate_decay_b)

* "manual"

  这是一种按已训练样本数分段取值的学习率退火方法。使用该learning_rate_schedule时，用户通过参数 :code:`learning_rate_args` 设置学习率衰减因子分段函数，当前的学习率为所设置 :code:`learning_rate` 与当前的衰减因子的乘积。以使用Adam算法为例，代码如下：

  ..  code-block:: python

      optimizer = paddle.optimizer.Adam(
          learning_rate=1e-3,
          learning_rate_schedule="manual",
          learning_rate_args="1000:1.0,2000:0.9,3000:0.8",)

  在该示例中，当已训练样本数小于等于1000时，学习率为 :code:`1e-3 * 1.0`；当已训练样本数大于1000小于等于2000时，学习率为 :code:`1e-3 * 0.9`；当已训练样本数大于2000时，学习率为 :code:`1e-3 * 0.8`。

* "pass_manual"

  这是一种按已训练pass数分段取值的学习率退火方法。使用该learning_rate_schedule时，用户通过参数 :code:`learning_rate_args` 设置学习率衰减因子分段函数，当前的学习率为所设置 :code:`learning_rate` 与当前的衰减因子的乘积。以使用Adam算法为例，代码如下：

  ..  code-block:: python

      optimizer = paddle.optimizer.Adam(
          learning_rate=1e-3,
          learning_rate_schedule="manual",
          learning_rate_args="1:1.0,2:0.9,3:0.8",)

  在该示例中，当已训练pass数小于等于1时，学习率为 :code:`1e-3 * 1.0`；当已训练pass数大于1小于等于2时，学习率为 :code:`1e-3 * 0.9`；当已训练pass数大于2时，学习率为 :code:`1e-3 * 0.8`。

23. 出现 :code:`Duplicated layer name` 错误怎么办
--------------------------------------------------

出现该错误的原因一般是用户对不同layer的参数 :code:`name` 设置了相同的取值。遇到该错误时，先找出参数 :code:`name` 取值相同的layer，然后将这些layer的参数 :code:`name` 设置为不同的值。

24. PaddlePaddle V2 API中，调用infer接口时输出多个层的计算结果
--------------------------------------------------

用户在使用多个中间网络层进行预测时，需要先将指定的网络层进行拼接，并作为 :code:`paddle.inference.Inference` 接口中 :code:`output_layer` 属性的输入, 然后调用infer接口来获取多个层对应的计算结果。 示例代码如下：

..      code-block:: bash

    inferer = paddle.inference.Inference(output_layer=[layer1, layer2],
                                        parameters=parameters)
    probs = inferer.infer(input=test_batch, field=["value"])

这里需要注意的是：

* 如果指定了2个layer作为输出层，实际上需要的输出结果是两个矩阵；
* 假设第一个layer的输出A是一个 N1 * M1 的矩阵，第二个 Layer 的输出B是一个 N2 * M2 的矩阵；
* paddle.v2 默认会将A和B 横向拼接，当N1 和 N2 大小不一样时，会报如下的错误：

..      code-block:: python

    ValueError: all the input array dimensions except for the concatenation axis must match exactly

多个层的输出矩阵的高度不一致，这种情况常常发生在：

* 同时输出序列层和非序列层；
* 多个输出层处理多个不同长度的序列;

此时可以在调用infer接口时通过设置 :code:`flatten_result=False` , 跳过“拼接”步骤，来解决上面的问题。这时，infer接口的返回值是一个python list:

* list元素的个数等于网络中输出层的个数；
* list 中每个元素是一个layer的输出结果矩阵，类型是numpy的ndarray；
* 每一个layer输出矩阵的高度，在非序列输入时：等于样本数；序列输入时等于：输入序列中元素的总数；宽度等于配置中layer的size；

25. PaddlePaddle 中不同的 recurrent layer 之间的差异
--------------------------------------------------
以LSTM为例，在PaddlePaddle中包含以下 recurrent layer：

* :code:`paddle.layer.lstmemory`
* :code:`paddle.networks.simple_lstm`
* :code:`paddle.networks.lstmemory_group`
* :code:`paddle.networks.bidirectional_lstm`

上述不同的recurrent layer可以归纳为2类：

* 由recurrent_group实现的recurrent layer：

  * 用户在使用这一类recurrent layer时，可以访问由recurrent unit在一个time step里计算得到的中间值（例如：hidden states, input-to-hidden mapping, memory cells等）；
  * 上述的 :code:`paddle.networks.lstmemory_group` 是这一类的recurrent layer；

* 将recurrent layer作为一个整体来实现：

  * 用户在使用这一类recurrent layer，只能访问它们的输出值；
  * 上述的 :code:`paddle.networks.lstmemory_group` ， :code:`paddle.networks.simple_lstm` 和 :code:`paddle.networks.bidirectional_lstm` 是这一类的recurrent layer；

在第一类recurrent layer的实现中，recurrent_group中包含许多基础layer的计算（例如：add, element-wise multiplication和matrix multiplication等），计算较为繁琐，而第二类的实现将recurrent layer作为一个整体，针对CPU和GPU计算做了更多优化。 所以，在实际应用中，第二类recurrent layer计算效率更高。 如果用户不需要访问LSTM的中间变量（例如：hidden states, input-to-hidden mapping, memory cells等），而只需要recurrent layer计算的输出，我们建议使用第二类recurrent layer。

除此之外，关于LSTM, PaddlePaddle中还包含 :code:`paddle.networks.lstmemory_unit` 这一计算单元：

  * 不同于上述介绍的recurrent layer , :code:`paddle.networks.lstmemory_unit` 定义了LSTM单元在一个time step里的计算过程，它并不是一个完整的recurrent layer，也不能接收序列数据作为输入；
  * :code:`paddle.networks.lstmemory_unit` 只能在recurrent_group中作为step function使用；

在LSTM和GRU中，隐状态的计算需要将输入数据进行线性映射（input-to-hidden mapping）。 在PaddlePaddle中，并不是所有的recurrent layer都将 input-to-hidden mapping 操作放在recurrent layer外面来执行来提升LSTM和GRU单元的计算速度。以 :code:`paddle.layer.lstmemory` 和 :code:`paddle.networks.simple_lstm` 为例：

  * :code:`paddle.layer.lstmemory` 内部不包含 input-to-hidden mapping 操作， 所以它并不是 `原有文献 <https://arxiv.org/abs/1308.0850>`_ 定义的LSTM完整形式；
  * 而 :code:`paddle.networks.simple_lstm` 中包含input-to-hidden mapping 操作，并结合 :code:`paddle.layer.lstmemory` 定义了完整的LSTM形式；

需要注意的是， :code:`paddle.networks.simple_lstm` 和 :code:`paddle.layer.lstmemory` 中定义的LSTM形式都包含了peephole connections，这也使得它们比不包含peephole connections的LSTM实现拥有更多的参数。
