快速开始
========

快速安装
--------

PaddlePaddle支持使用pip快速安装，目前支持CentOS 6以上, Ubuntu 14.04以及MacOS 10.12，并安装有Python2.7。
执行下面的命令完成快速安装，版本为cpu_avx_openblas：

  .. code-block:: bash

     pip install paddlepaddle

如果需要安装支持GPU的版本（cuda7.5_cudnn5_avx_openblas），需要执行：

  .. code-block:: bash

     pip install paddlepaddle-gpu

更详细的安装和编译方法参考：:ref:`install_steps` 。

快速使用
--------

创建一个 housing.py 并粘贴此Python代码：

  .. code-block:: python

     import paddle.v2 as paddle

     # Initialize PaddlePaddle.
     paddle.init(use_gpu=False, trainer_count=1)

     # Configure the neural network.
     x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
     y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())

     # Infer using provided test data.
     probs = paddle.infer(
         output_layer=y_predict,
         parameters=paddle.dataset.uci_housing.model(),
         input=[item for item in paddle.dataset.uci_housing.test()()])

     for i in xrange(len(probs)):
         print 'Predicted price: ${:,.2f}'.format(probs[i][0] * 1000)

执行 :code:`python housing.py` 瞧！ 它应该打印出预测住房数据的清单。
