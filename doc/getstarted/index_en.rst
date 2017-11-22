GET STARTED
============

.. _quick_install:

Quick Install
----------------------

You can use pip to install PaddlePaddle using a single command, supports
CentOS 6 above, Ubuntu 14.04 above or MacOS 10.12, with Python 2.7 installed.
Simply run the following command to install:

  .. code-block:: bash

     pip install paddlepaddle

If you need to install GPU version, run:

  .. code-block:: bash

     pip install paddlepaddle-gpu

For more details about installation and build:

..  toctree::
  :maxdepth: 1

  build_and_install/index_en.rst


.. _quick_start:

Quick Start
++++++++

Download the `trained housing prices model <https://raw.githubusercontent.com/PaddlePaddle/book/develop/01.fit_a_line/fit_a_line.tar>`_

Now, create a new file called housing.py, and paste this Python
code (make sure to set the right path based on the location of fit_a_line.tar
on your computer):


  .. code-block:: python

     import paddle.v2 as paddle

     # Initialize PaddlePaddle.
     paddle.init(use_gpu=False, trainer_count=1)

     # Configure the neural network.
     x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
     y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())

     with open('fit_a_line.tar', 'r') as f:
         parameters = paddle.parameters.Parameters.from_tar(f)

     # Infer using provided test data.
     probs = paddle.infer(
          output_layer=y_predict, parameters=parameters,
          input=[item for item in paddle.dataset.uci_housing.test()()])

     for i in xrange(len(probs)):
          print 'Predicted price: ${:,.2f}'.format(probs[i][0] * 1000)

Run :code:`python housing.py` and voila! It should print out a list of predictions
for the test housing data.
