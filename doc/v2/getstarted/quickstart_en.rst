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

For more details about installation and build: :ref:`install_steps` .

Quick Use
---------

Create a new file called housing.py, and paste this Python
code:


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

Run :code:`python housing.py` and voila! It should print out a list of predictions
for the test housing data.
