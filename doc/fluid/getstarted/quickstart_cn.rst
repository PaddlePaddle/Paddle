快速开始
========

快速安装
--------

PaddlePaddle支持使用pip快速安装，目前支持CentOS 6以上, Ubuntu 14.04以及MacOS 10.12，并安装有Python2.7。
执行下面的命令完成快速安装，版本为cpu_avx_openblas：

  .. code-block:: bash

     pip install paddlepaddle

如果需要安装支持GPU的版本（cuda8.0_cudnn5_avx_openblas），需要执行：

  .. code-block:: bash

     pip install paddlepaddle-gpu

更详细的安装和编译方法参考： :ref:`install_steps` 。

快速使用
--------

创建一个 housing.py 并粘贴此Python代码：

  .. code-block:: python

     import paddle.dataset.uci_housing as uci_housing
     import paddle.fluid as fluid

     with fluid.scope_guard(fluid.core.Scope()):
         # initialize executor with cpu
         exe = fluid.Executor(place=fluid.CPUPlace())
         # load inference model
         [inference_program, feed_target_names,fetch_targets] =  \
             fluid.io.load_inference_model(uci_housing.fluid_model(), exe)
         # run inference
         result = exe.run(inference_program,
                          feed={feed_target_names[0]: uci_housing.predict_reader()},
                          fetch_list=fetch_targets)
         # print predicted price is $12,273.97
         print 'Predicted price: ${:,.2f}'.format(result[0][0][0] * 1000)

执行 :code:`python housing.py` 瞧！ 它应该打印出预测住房数据的清单。
