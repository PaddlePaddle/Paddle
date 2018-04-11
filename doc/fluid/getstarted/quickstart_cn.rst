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
     import paddle
     import paddle.fluid as fluid
     
     
     x = fluid.layers.data(name='x', shape=[13], dtype='float32')
     place = fluid.CPUPlace()
     exe = fluid.Executor(place=place)
     feeder = fluid.DataFeeder(place=place, feed_list=[x])
     
     with fluid.scope_guard(fluid.core.Scope()):
         parameter_model = paddle.dataset.uci_housing.fluid_model()
     
         [inference_program, feed_target_names,fetch_targets] =  \
             fluid.io.load_inference_model(parameter_model, exe)
     
         predict_reader = paddle.batch(paddle.dataset.uci_housing.predict_reader(), batch_size=20)
     
         results = []
         for data in predict_reader():
             result = exe.run(inference_program,
                               feed=feeder.feed(data),
                               fetch_list=fetch_targets)
             results.append(result)
     
         for res in results:
             for i in xrange(len(res[0])):
                 print 'Predicted price: ${:,.2f}'.format(res[0][i][0] * 1000)
执行 :code:`python housing.py` 瞧！ 它应该打印出预测住房数据的清单。
