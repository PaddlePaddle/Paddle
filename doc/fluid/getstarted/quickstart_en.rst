Quick Start
============

Quick Install
-------------

You can use pip to install PaddlePaddle with a single command, supports
CentOS 6 above, Ubuntu 14.04 above or MacOS 10.12, with Python 2.7 installed.
Simply run the following command to install, the version is cpu_avx_openblas:

  .. code-block:: bash

     pip install paddlepaddle

If you need to install GPU version (cuda7.5_cudnn5_avx_openblas), run:

  .. code-block:: bash

     pip install paddlepaddle-gpu

For more details about installation and build: `install and Compile <http://www.paddlepaddle.org/docs/develop/documentation/fluid/en/build_and_install/index_en.html>`_ .

Quick Use
---------

Create a new file called housing.py, and paste this Python
code:


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

Run :code:`python housing.py` and voila! It should print out a list of predictions
for the test housing data.
