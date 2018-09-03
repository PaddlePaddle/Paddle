Quick Start
============

Quick Install
-------------

You can use pip to install PaddlePaddle with a single command, supports
CentOS 6 above, Ubuntu 14.04 above or MacOS 10.12, with Python 2.7 installed.
Simply run the following command to install, the version is cpu_avx_openblas:

  .. code-block:: bash

     pip install paddlepaddle

If you need to install GPU version (cuda8.0_cudnn5_avx_openblas), run:

  .. code-block:: bash

     pip install paddlepaddle-gpu

For more details about installation and build: :ref:`install_steps` .

Quick Use
---------

Create a new file called housing.py, and paste this Python
code:


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

Run :code:`python housing.py` and voila! It should print out a list of predictions
for the test housing data.
