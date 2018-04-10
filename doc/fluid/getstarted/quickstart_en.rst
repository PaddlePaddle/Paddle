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
    import sys
    
    import math
    import numpy
    
    import paddle.fluid as fluid
    import paddle.fluid.core as core
    import paddle
    
    def train(save_dirname):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)
    
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)
    
        BATCH_SIZE = 20
    
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500), batch_size=BATCH_SIZE)
    
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
    
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(fluid.default_startup_program())
    
        main_program = fluid.default_main_program()
    
        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_loss_value, = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost])
                if avg_loss_value[0] < 10.0:
                    if save_dirname is not None:
                        fluid.io.save_inference_model(save_dirname, ['x'],
                                                      [y_predict], exe)
                    return
                if math.isnan(float(avg_loss_value)):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError("Fit a line cost is too large, {0:2.2}".format(
            avg_loss_value[0]))
    
    def infer(save_dirname):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
    
        probs = []
    
        inference_scope = fluid.core.Scope()
        with fluid.scope_guard(inference_scope):
            # Use fluid.io.load_inference_model to obtain the inference program desc,
            # the feed_target_names (the names of variables that will be feeded
            # data using feed operators), and the fetch_targets (variables that
            # we want to obtain data from using fetch operators).
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
    
            # The input's dimension should be 2-D and the second dim is 13
            # The input data should be >= 0
            batch_size = 10
            tensor_x = numpy.random.uniform(0, 10,
                                            [batch_size, 13]).astype("float32")
            assert feed_target_names[0] == 'x'
            results = exe.run(inference_program,
                              feed={feed_target_names[0]: tensor_x},
                              fetch_list=fetch_targets)
            probs.append(results)
    
        for i in xrange(len(probs)):
            print(probs[i][0] * 1000)
            print('Predicted price: ${0}'.format(probs[i][0] * 1000))
    
    def main():
        # Directory for saving the trained model
        save_dirname = "fit_a_line.inference.model"
    
        train(save_dirname)
        infer(save_dirname)
    
    if __name__=="__main__":
        main()
Run :code:`python housing.py` and voila! It should print out a list of predictions
for the test housing data.
