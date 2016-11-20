PaddlePaddle快速入门教程
========================

我们将以 `文本分类问题 <https://en.wikipedia.org/wiki/Document_classification>`_ 为例,
介绍PaddlePaddle的基本使用方法。

安装
====

请参考 `安装教程 <../../build_and_install/index.html>`_ 安装PaddlePaddle。

使用概述
========

**文本分类问题**：对于给定的一条文本，我们从提前给定的类别集合中选择其所属类别。

比如, 在购物网站上，通过查看买家对某个产品的评价反馈, 评估该产品的质量。

- 这个显示器很棒！ （好评）
- 用了两个月之后这个显示器屏幕碎了。（差评）

使用PaddlePaddle, 每一个任务流程都可以被划分为如下五个步骤。

    ..  image:: Pipeline.jpg
        :align: center
        :scale: 80%

1. 数据格式准备
    - 本例每行保存一条样本，类别Id和文本信息用 ``Tab`` 间隔，文本中的单词用空格分隔（如果不切词，则字与字之间用空格分隔），例如：``类别Id '\t' 这 个 显 示 器 很 棒 ！``
2. 向系统传送数据
    - PaddlePaddle可以执行用户的python脚本程序来读取各种格式的数据文件。
    - 本例的所有字符都将转换为连续整数表示的Id传给模型。
3. 描述网络结构和优化算法
    - 本例由易到难展示4种不同的文本分类网络配置：逻辑回归模型，词向量模型，卷积模型，时序模型。
    - 常用优化算法包括Momentum, RMSProp，AdaDelta，AdaGrad，Adam，Adamax等，本例采用Adam优化方法，加了L2正则和梯度截断。
4. 训练模型
5. 应用模型

数据格式准备
------------

接下来我们将展示如何用PaddlePaddle训练一个文本分类模型，将 `Amazon电子产品评论数据 <http://jmcauley.ucsd.edu/data/amazon/>`_ 分为好评(正样本)和差评(负样本)两种类别。
`源代码 <https://github.com/baidu/Paddle>`_ 的 ``demo/quick_start`` 目录里提供了该数据的下载脚本和预处理脚本。

.. code-block:: bash

    cd demo/quick_start
    ./data/get_data.sh
    ./preprocess.sh

向系统传送数据
==============

Python数据读取脚本
------------------

`DataProvider <../../ui/data_provider/index.html>`_ 是PaddlePaddle负责提供数据的模块。``DataProvider`` 主要职责在于将训练数据传入内存或者显存，让模型能够得到训练更新，其包括两个函数：

* initializer：PaddlePaddle会在调用读取数据的Python脚本之前，先调用initializer函数。在下面例子里，我们在initialzier函数里初始化词表，并且在随后的读取数据过程中填充词表。
* process：PaddlePaddle调用process函数来读取数据。每次读取一条数据后，process函数会用yield语句输出这条数据，从而能够被PaddlePaddle 捕获 (harvest)。

``dataprovider_bow.py`` 文件给出了完整例子：

.. code-block:: python

    from paddle.trainer.PyDataProvider2 import *

    # id of the word not in dictionary
    UNK_IDX = 0

    # initializer is called by the framework during initialization.
    # It allows the user to describe the data types and setup the
    # necessary data structure for later use.
    # `settings` is an object. initializer need to properly fill settings.input_types.
    # initializer can also store other data structures needed to be used at process().
    # In this example, dictionary is stored in settings.
    # `dictionay` and `kwargs` are arguments passed from trainer_config.lr.py
    def initializer(settings, dictionary, **kwargs):
        # Put the word dictionary into settings
        settings.word_dict = dictionary

        # setting.input_types specifies what the data types the data provider
        # generates.
        settings.input_types = [
            # The first input is a sparse_binary_vector,
            # which means each dimension of the vector is either 0 or 1. It is the
            # bag-of-words (BOW) representation of the texts.
            sparse_binary_vector(len(dictionary)),
            # The second input is an integer. It represents the category id of the
            # sample. 2 means there are two labels in the dataset.
            # (1 for positive and 0 for negative)
            integer_value(2)]

    # Delaring a data provider. It has an initializer 'data_initialzer'.
    # It will cache the generated data of the first pass in memory, so that
    # during later pass, no on-the-fly data generation will be needed.
    # `setting` is the same object used by initializer()
    # `file_name` is the name of a file listed train_list or test_list file given
    # to define_py_data_sources2(). See trainer_config.lr.py.
    @provider(init_hook=initializer, cache=CacheType.CACHE_PASS_IN_MEM)
    def process(settings, file_name):
        # Open the input data file.
        with open(file_name, 'r') as f:
            # Read each line.
            for line in f:
                # Each line contains the label and text of the comment, separated by \t.
                label, comment = line.strip().split('\t')

                # Split the words into a list.
                words = comment.split()

                # convert the words into a list of ids by looking them up in word_dict.
                word_vector = [settings.word_dict.get(w, UNK_IDX) for w in words]

                # Return the features for the current comment. The first is a list
                # of ids representing a 0-1 binary sparse vector of the text,
                # the second is the integer id of the label.
                yield word_vector, int(label)

配置中的数据加载定义
--------------------

在模型配置中通过 ``define_py_data_sources2`` 接口来加载数据：

.. code-block:: python

    from paddle.trainer_config_helpers import *

    file = "data/dict.txt"
    word_dict = dict()
    with open(dict_file, 'r') as f:
        for i, line in enumerate(f):
            w = line.strip().split()[0]
            word_dict[w] = i
    # define the data sources for the model.
    # We need to use different process for training and prediction.
    # For training, the input data includes both word IDs and labels.
    # For prediction, the input data only includs word Ids.
    define_py_data_sources2(train_list='data/train.list',
                            test_list='data/test.list',
                            module="dataprovider_bow",
                            obj="process",
                            args={"dictionary": word_dict})


以下是对上述数据加载的解释：

- data/train.list,data/test.list: 指定训练数据和测试数据
- module="dataprovider_bow": 数据处理的Python脚本文件名
- obj="process": 指定生成数据的函数
- args={"dictionary": word_dict}: 额外的参数，这里指定词典

更详细数据格式和用例请参考 `PyDataProvider2 <../../ui/data_provider/pydataprovider2.html>`_ 。

模型网络结构
============

本小节我们将介绍模型网络结构。

    ..  image:: PipelineNetwork.jpg
        :align: center
        :scale: 80%


我们将以基本的逻辑回归网络作为起点，并逐渐展示更加深入的功能。更详细的网络配置连接请参考 `Layer文档 <../../../doc/layer.html>`_ 。
所有配置都在 `源代码 <https://github.com/baidu/Paddle>`_ 的 ``demo/quick_start`` 目录下。

逻辑回归模型
------------

具体流程如下:

    ..  image:: NetLR.jpg
        :align: center
        :scale: 80%

- 获取利用one-hot vector表示的每个单词，维度是词典大小

    .. code-block:: python

        word = data_layer(name="word",  size=word_dim)

- 获取该条样本类别Id，维度是类别个数。

    .. code-block:: python

        label = data_layer(name="label", size=label_dim)

- 利用逻辑回归模型对该向量进行分类，同时会计算分类准确率

    .. code-block:: python

        # Define a fully connected layer with logistic activation (also called softmax activation).
        output = fc_layer(input=word,
                        size=label_dim,
                        act_type=SoftmaxActivation())
        # Define cross-entropy classification loss and error.
        classification_cost(input=output, label=label)


 - input: 除过data层，每个层都有一个或多个input,多个input以list方式输入
 - size: 该层神经元个数
 - act_type: 激活函数类型

**效果总结**：我们将在后面介绍训练和预测流程的脚本。在此为方便对比不同网络结构，我们总结了各个网络的复杂度和效果。

    =====================  ===============================  =================
    网络名称	                    参数数量                    错误率
    =====================  ===============================  =================
    逻辑回归	                  252 KB                       8.652 %
    =====================  ===============================  =================

词向量模型
----------

embedding模型需要稍微改变数据提供的脚本，即 ``dataprovider_emb.py``，词向量模型、
卷积模型、时序模型均使用该脚本。其中文本输入类型定义为整数时序类型integer_value_sequence。

.. code-block:: python

    def initializer(settings, dictionary, **kwargs):
        settings.word_dict = dictionary
        settings.input_types = [
            # Define the type of the first input as sequence of integer.
            # The value of the integers range from 0 to len(dictrionary)-1
            integer_value_sequence(len(dictionary)),
            # Define the second input for label id
            integer_value(2)]

    @provider(init_hook=initializer)
    def process(settings, file_name):
        ...
        # omitted, it is same as the data provider for LR model

该模型依然是使用逻辑回归分类网络的框架， 只是将句子利用连续向量表示替换稀疏
向量表示， 即对第3步进行替换。句子表示的计算更新为2步：

..  image:: NetContinuous.jpg
    :align: center
    :scale: 80%

- 利用单词Id查找对应的该单词的连续表示向量(维度为word_dim)， 输入N个单词，输出为N个word_dim维度向量

    .. code-block:: python

        emb = embedding_layer(input=word, size=word_dim)

- 将该句话包含的所有单词向量求平均得到句子的表示

    .. code-block:: python

        avg = pooling_layer(input=emb, pooling_type=AvgPooling())

其它部分和逻辑回归网络结构一致。

**效果总结：**

    =====================  ===============================  ==================
    网络名称	                    参数数量                    错误率
    =====================  ===============================  ==================
    词向量模型	                  15 MB                       8.484 %
    =====================  ===============================  ==================

卷积模型
-----------

卷积网络是一种特殊的从词向量表示到句子表示的方法， 也就是将词向量模型额步
骤3-2进行进一步演化， 变为3个新的子步骤。

..  image:: NetConv.jpg
    :align: center
    :scale: 80%

文本卷积分为三个步骤:

1. 获取每个单词左右各k个近邻， 拼接成一个新的向量表示；

2. 对该表示进行非线性变换 （例如Sigmoid变换）, 成为维度为hidden_dim的新的向量；

3. 在每个维度上取出在该句话新的向量集合上该维度的最大值作为最后的句子表示向量。 这3个子步骤可配置为:

.. code-block:: python

    text_conv = sequence_conv_pool(input=emb,
                                context_start=k,
                                context_len=2 * k + 1)

**效果总结：**

    =====================  ===============================  ========================
    网络名称	                    参数数量                    错误率
    =====================  ===============================  ========================
    卷积模型	                  16 MB                       5.628 %
    =====================  ===============================  ========================

时序模型
----------

..  image:: NetRNN.jpg
    :align: center
    :scale: 80%

时序模型即为RNN模型, 包括简单的RNN模型、GRU模型、LSTM模型等。

- GRU模型配置：

    .. code-block:: python

        gru = simple_gru(input=emb, size=gru_size)


- LSTM模型配置：

    .. code-block:: python

        lstm = simple_lstm(input=emb, size=lstm_size)

针对本问题，我们采用单层LSTM模型，并使用了Dropout，**效果总结：**

    =====================  ===============================  =========================
    网络名称	                    参数数量                    错误率
    =====================  ===============================  =========================
    时序模型	                  16 MB                       4.812 %
    =====================  ===============================  =========================

优化算法
=========

`优化算法 <../../../doc/ui/trainer_config_helpers_api.html#module-paddle.trainer_config_helpers.optimizers>`_ 包括
Momentum, RMSProp，AdaDelta，AdaGrad，ADAM，Adamax等，这里采用Adam优化方法，加了L2正则和梯度截断。

.. code-block:: python

    settings(batch_size=128,
            learning_rate=2e-3,
            learning_method=AdamOptimizer(),
            regularization=L2Regularization(8e-4),
            gradient_clipping_threshold=25)

训练模型
=========

在完成了数据和网络结构搭建之后， 我们进入到训练部分。

..  image:: PipelineTrain.jpg
    :align: center
    :scale: 80%

训练脚本：我们将训练的命令行保存在了 ``train.sh`` 文件中。训练时所需设置的主要参数如下：

    .. code-block:: bash

        paddle train \
        --config=trainer_config.py \
        --log_period=20 \
        --save_dir=./output \
        --num_passes=15 \
        --use_gpu=false

这里没有介绍多机分布式训练，可以参考 `分布式训练 <../../cluster/index.html>`_ 的demo学习如何进行多机训练。

预测
=====

可以使用训练好的模型评估带有label的验证集，也可以预测没有label的测试集。

..  image:: PipelineTest.jpg
    :align: center
    :scale: 80%

测试脚本如下，将会测试配置文件中test.list指定的数据。

    .. code-block:: bash

        paddle train \
        --use_gpu=false \
        --job=test \
        --init_model_path=./output/pass-0000x

可以参考 `Python API预测 <../../ui/predict/swig_py_paddle.html>`_
教程，或其他 `demo <../../demo/index.html>`_ 的Python预测过程。也可以通过如下方式预测。

预测脚本(``predict.sh``)：

    .. code-block:: bash

        model="output/pass-00003"
        paddle train \
            --config=trainer_config.lstm.py \
            --use_gpu=false \
            --job=test \
            --init_model_path=$model \
            --config_args=is_predict=1 \
            --predict_output_dir=. \

        mv rank-00000 result.txt

这里以 ``output/pass-00003`` 为例进行预测，用户可以根据训练log选择test结果最好的模型来预测。与训练网络配置不同的是：无需label相关的层，指定outputs输出概率层(softmax输出)，
指定batch_size=1，数据传输无需label数据，预测数据指定test_list的位置。

预测结果以文本的形式保存在 ``result.txt`` 中，一行为一个样本，格式如下：

    .. code-block:: bash

        预测ID;ID为0的概率 ID为1的概率
        预测ID;ID为0的概率 ID为1的概率

    .. code-block:: python

        is_predict = get_config_arg('is_predict', bool, False)
        trn = 'data/train.list' if not is_predict else None
        tst = 'data/test.list' if not is_predict else 'data/pred.list'
        obj = 'process' if not is_predict else 'process_pre'
        batch_size = 128 if not is_predict else 1
        if is_predict:
            maxid = maxid_layer(output)
            outputs([maxid,output])
        else:
            label = data_layer(name="label", size=2)
            cls = classification_cost(input=output, label=label)
            outputs(cls)

总体效果总结
==============

这些流程中的数据下载、网络配置、训练脚本在 ``/demo/quick_start`` 目录，我们在此总
结上述网络结构在Amazon-Elec测试集(25k)上的效果:

    =====================  ===============================  =============  ==================================
    网络名称	                    参数数量                    错误率          配置文件
    =====================  ===============================  =============  ==================================
    逻辑回归模型	                  252 KB                     8.652%          trainer_config.lr.py
    词向量模型      	               15 MB                      8.484%         trainer_config.emb.py
    卷积模型                        16 MB                     5.628%          trainer_config.cnn.py
    时序模型 	                    16 MB                     4.812%          trainer_config.lstm.py
    =====================  ===============================  =============  ==================================


附录
=====

命令行参数
----------

* \--config：网络配置
* \--save_dir：模型存储路径
* \--log_period：每隔多少batch打印一次日志
* \--num_passes：训练轮次，一个pass表示过一遍所有训练样本
* \--config_args：命令指定的参数会传入网络配置中。
* \--init_model_path：指定初始化模型路径，可用在测试或训练时指定初始化模型。

默认一个pass保存一次模型，也可以通过saving_period_by_batches设置每隔多少batch保存一次模型。
可以通过show_parameter_stats_period设置打印参数信息等。
其他参数请参考 `命令行参数文档 <../../ui/index.html#command-line-argument>`_ 。

输出日志
---------

.. code-block:: bash

    TrainerInternal.cpp:160]  Batch=20 samples=2560 AvgCost=0.628761 CurrentCost=0.628761 Eval: classification_error_evaluator=0.304297  CurrentEval: classification_error_evaluator=0.304297

模型训练会看到这样的日志，详细的参数解释如下面表格：

    ===========================================  ========================================================== 
    名称	                                         解释
    ===========================================  ==========================================================
    Batch=20	                                  表示过了20个batch
    samples=2560	                              表示过了2560个样本
    AvgCost	                                      每个pass的第0个batch到当前batch所有样本的平均cost
    CurrentCost	                                  当前log_period个batch所有样本的平均cost
    Eval: classification_error_evaluator	      每个pass的第0个batch到当前batch所有样本的平均分类错误率
    CurrentEval: classification_error_evaluator	  当前log_period个batch所有样本的平均分类错误率
    ===========================================  ==========================================================
