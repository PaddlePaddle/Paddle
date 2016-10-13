PaddlePaddle的Python预测接口
==================================

PaddlePaddle目前使用Swig对其常用的预测接口进行了封装，使在Python环境下的预测接口更加简单。
在Python环境下预测结果，主要分为以下几个步骤。

* 读入解析训练配置
* 构造GradientMachine
* 准备数据
* 预测

典型的预测代码如下，使用mnist手写识别作为样例, 完整代码见
:code:`src_root/doc/ui/predict/predict_sample.py` 。

..  literalinclude:: ../../../doc/ui/predict/predict_sample.py
    :language: python
    :lines: 15-18,90-100,101-104

主要的软件包为py_paddle.swig_paddle，这个软件包文档相对完善。可以使用python的
:code:`help()` 函数查询文档。主要步骤为:

* 在程序开始阶段，使用 :code:`swig_paddle.initPaddle()` 传入命令行参数初始化
  PaddlePaddle。详细的命令行参数请参考
  `命令行参数 <../cmd_argument/detail_introduction.html>`_ 。
* 接下来使用 :code:`parse_config()` 解析训练时的配置文件。这里要注意预测数据通常
  不包含label, 而且预测网络通常直接输出最后一层的结果而不是像训练时一样以cost
  layer作为输出，所以用于预测的配置文件要做相应的修改。
* 使用 :code:`swig_paddle.GradientMachine.createFromConfigproto()` 根据上一步解
  析好的配置创建神经网络。
* 创建一个 :code:`DataProviderConverter` 对象converter。
    - swig_paddle接受的原始数据是C++的Matrix，也就是直接写内存的float数组。
      这个接口并不用户友好。所以，我们提供了一个工具类DataProviderConverter。
      这个工具类接收和PyDataProvider2一样的输入数据，详情请参考
      `PyDataProvider2文档 <../../../doc/ui/data_provider/pydataprovider2.html>`_ 。
* 最后使用 :code:`forwardTest()` 直接提取出神经网络Output层的输出结果。典型的输出结果为\:

..  code-block:: text

    [{'id': None, 'value': array([[  5.53018653e-09,   1.12194102e-05,   1.96644767e-09,
          1.43630644e-02,   1.51111044e-13,   9.85625684e-01,
          2.08823112e-10,   2.32777140e-08,   2.00186201e-09,
          1.15501715e-08],
       [  9.99982715e-01,   1.27787406e-10,   1.72296313e-05,
          1.49316648e-09,   1.36540484e-11,   6.93137714e-10,
          2.70634608e-08,   3.48565123e-08,   5.25639710e-09,
          4.48684503e-08]], dtype=float32)}]

其中，value即为softmax层的输出。由于数据是两条，所以输出的value包含两个向量 。
