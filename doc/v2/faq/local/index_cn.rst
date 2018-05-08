###############
本地训练与预测
###############

..  contents::

1. 如何减少内存占用
-------------------

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

2. 如何加速训练速度
-------------------

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

利用更多的计算资源可以分为以下几个方式来进行\:

* 单机CPU训练

  * 使用多线程训练。设置命令行参数 :code:`trainer_count`。

* 单机GPU训练

  * 使用显卡训练。设置命令行参数 :code:`use_gpu`。
  * 使用多块显卡训练。设置命令行参数 :code:`use_gpu` 和 :code:`trainer_count` 。

* 多机训练

  * 请参考 :ref:`cluster_train` 。

3. 如何指定GPU设备
------------------

例如机器上有4块GPU，编号从0开始，指定使用2、3号GPU：

* 方式1：通过 `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ 环境变量来指定特定的GPU。

..      code-block:: bash

        env CUDA_VISIBLE_DEVICES=2,3 paddle train --use_gpu=true --trainer_count=2

* 方式2：通过命令行参数 ``--gpu_id`` 指定。

..      code-block:: bash

        paddle train --use_gpu=true --trainer_count=2 --gpu_id=2


4. 训练过程中出现 :code:`Floating point exception`, 训练因此退出怎么办?
------------------------------------------------------------------------

Paddle二进制在运行时捕获了浮点数异常，只要出现浮点数异常(即训练过程中出现NaN或者Inf)，立刻退出。浮点异常通常的原因是浮点数溢出、除零等问题。
主要原因包括两个方面:

* 训练过程中参数或者训练过程中的梯度尺度过大，导致参数累加，乘除等时候，导致了浮点数溢出。
* 模型一直不收敛，发散到了一个数值特别大的地方。
* 训练数据有问题，导致参数收敛到了一些奇异的情况。或者输入数据尺度过大，有些特征的取值达到数百万，这时进行矩阵乘法运算就可能导致浮点数溢出。

这里有两种有效的解决方法：

1. 设置 :code:`gradient_clipping_threshold` 参数，示例代码如下：

..  code-block:: python

    optimizer = paddle.optimizer.RMSProp(
        learning_rate=1e-3,
        gradient_clipping_threshold=10.0,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4))

具体可以参考  `nmt_without_attention  <https://github.com/PaddlePaddle/models/blob/develop/nmt_without_attention/train.py#L35>`_ 示例。

2. 设置 :code:`error_clipping_threshold` 参数，示例代码如下：

..  code-block:: python

    decoder_inputs = paddle.layer.fc(
        act=paddle.activation.Linear(),
        size=decoder_size * 3,
        bias_attr=False,
        input=[context, current_word],
        layer_attr=paddle.attr.ExtraLayerAttribute(
            error_clipping_threshold=100.0))

完整代码可以参考示例 `machine translation <https://github.com/PaddlePaddle/book/blob/develop/08.machine_translation/train.py#L66>`_ 。

两种方法的区别：

1. 两者都是对梯度的截断，但截断时机不同，前者在 :code:`optimzier` 更新网络参数时应用；后者在激活函数反向计算时被调用；
2. 截断对象不同：前者截断可学习参数的梯度，后者截断回传给前层的梯度;

除此之外，还可以通过减小学习率或者对数据进行归一化处理来解决这类问题。

5.  如何调用 infer 接口输出多个layer的预测结果
-----------------------------------------------

* 将需要输出的层作为 :code:`paddle.inference.Inference()` 接口的 :code:`output_layer` 参数输入，代码如下：

..  code-block:: python

    inferer = paddle.inference.Inference(output_layer=[layer1, layer2], parameters=parameters)

* 指定要输出的字段进行输出。以输出 :code:`value` 字段为例，代码如下：

..  code-block:: python

    out = inferer.infer(input=data_batch, field=["value"])

需要注意的是：

* 如果指定了2个layer作为输出层，实际上需要的输出结果是两个矩阵；
* 假设第一个layer的输出A是一个 N1 * M1 的矩阵，第二个 Layer 的输出B是一个 N2 * M2 的矩阵；
* paddle.v2 默认会将A和B 横向拼接，当N1 和 N2 大小不一样时，会报如下的错误：

..      code-block:: python

    ValueError: all the input array dimensions except for the concatenation axis must match exactly

多个层的输出矩阵的高度不一致导致拼接失败，这种情况常常发生在：

* 同时输出序列层和非序列层；
* 多个输出层处理多个不同长度的序列;

此时可以在调用infer接口时通过设置 :code:`flatten_result=False` , 跳过“拼接”步骤，来解决上面的问题。这时，infer接口的返回值是一个python list:

* list 中元素的个数等于网络中输出层的个数；
* list 中每个元素是一个layer的输出结果矩阵，类型是numpy的ndarray；
* 每一个layer输出矩阵的高度，在非序列输入时：等于样本数；序列输入时等于：输入序列中元素的总数；宽度等于配置中layer的size；

6.  如何在训练过程中获得某一个layer的output
-----------------------------------------------

可以在event_handler中，通过 :code:`event.gm.getLayerOutputs("layer_name")` 获得在模型配置中某一层的name :code:`layer_name` 在当前
mini-batch forward的output的值。获得的值类型均为 :code:`numpy.ndarray` ，可以通过这个输出来完成自定义的评估指标计算等功能。例如下面代码：

..      code-block:: python

        def score_diff(right_score, left_score):
            return np.average(np.abs(right_score - left_score))

        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 25 == 0:
                    diff = score_diff(
                        event.gm.getLayerOutputs("right_score")["right_score"][
                            "value"],
                        event.gm.getLayerOutputs("left_score")["left_score"][
                            "value"])
                    logger.info(("Pass %d Batch %d : Cost %.6f, "
                                "average absolute diff scores: %.6f") %
                                (event.pass_id, event.batch_id, event.cost, diff))

注意：此方法不能获取 :code:`paddle.layer.recurrent_group` 里step的内容，但可以获取 :code:`paddle.layer.recurrent_group` 的输出。

7.  如何在训练过程中获得参数的权重和梯度
-----------------------------------------------

在某些情况下，获得当前mini-batch的权重（或称作weights, parameters）有助于在训练时观察具体数值，方便排查以及快速定位问题。
可以通过在 :code:`event_handler` 中打印其值（注意，需要使用 :code:`paddle.event.EndForwardBackward` 保证使用GPU训练时也可以获得），
示例代码如下：

..      code-block:: python

        ...
        parameters = paddle.parameters.create(cost)
        ...
        def event_handler(event):
            if isinstance(event, paddle.event.EndForwardBackward):
                if event.batch_id % 25 == 0:
                    for p in parameters.keys():
                        logger.info("Param %s, Grad %s",
                            parameters.get(p), parameters.get_grad(p))

注意：“在训练过程中获得某一个layer的output”和“在训练过程中获得参数的权重和梯度”都会造成训练中的数据从C++拷贝到numpy，会对训练性能造成影响。不要在注重性能的训练场景下使用。