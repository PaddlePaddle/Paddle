.. _api_swig_py_paddle:

基于Python的预测
================

预测流程
--------

PaddlePaddle使用swig对常用的预测接口进行了封装，通过编译会生成py_paddle软件包，安装该软件包就可以在python环境下实现模型预测。可以使用python的 ``help()`` 函数查询软件包相关API说明。

基于Python的模型预测，主要包括以下五个步骤。

1. 初始化PaddlePaddle环境

   在程序开始阶段，通过调用 ``swig_paddle.initPaddle()`` 并传入相应的命令行参数初始化PaddlePaddle。

2. 解析模型配置文件
   
   初始化之后，可以通过调用 ``parse_config()`` 解析训练模型时用的配置文件。注意预测数据通常不包含label, 同时预测网络通常直接输出最后一层的结果而不是像训练网络一样再接一层cost layer，所以一般需要对训练用的模型配置文件稍作相应修改才能在预测时使用。

3. 构造paddle.GradientMachine
  
   通过调用 ``swig_paddle.GradientMachine.createFromConfigproto()`` 传入上一步解析出来的模型配置就可以创建一个 ``GradientMachine``。

4. 准备预测数据
  
   swig_paddle中的预测接口的参数是自定义的C++数据类型，py_paddle里面提供了一个工具类 ``DataProviderConverter`` 可以用于接收和PyDataProvider2一样的输入数据并转换成预测接口所需的数据类型。

5. 模型预测
  
   通过调用 ``forwardTest()`` 传入预测数据，直接返回计算结果。


预测Demo
--------

如下是一段使用mnist model来实现手写识别的预测代码。完整的代码见 ``src_root/doc/ui/predict/predict_sample.py`` 。mnist model可以通过 ``src_root\demo\mnist`` 目录下的demo训练出来。

..  literalinclude:: src/predict_sample.py
    :language: python
    :lines: 15-18,121-136


Demo预测输出如下，其中value即为softmax层的输出。由于TEST_DATA包含两条预测数据，所以输出的value包含两个向量 。

..  code-block:: text

    [{'id': None, 'value': array(
      [[  5.53018653e-09,   1.12194102e-05,   1.96644767e-09,
          1.43630644e-02,   1.51111044e-13,   9.85625684e-01,
          2.08823112e-10,   2.32777140e-08,   2.00186201e-09,
          1.15501715e-08],
       [  9.99982715e-01,   1.27787406e-10,   1.72296313e-05,
          1.49316648e-09,   1.36540484e-11,   6.93137714e-10,
          2.70634608e-08,   3.48565123e-08,   5.25639710e-09,
          4.48684503e-08]], dtype=float32)}]


