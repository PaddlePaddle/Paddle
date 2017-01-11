# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
An example to show how to use current Raw SWIG API to train cifar-10 network.

Current implementation uses Raw SWIG, which means the API call is directly \
passed to C++ side of Paddle.

The user api should be simpler and carefully designed.
"""
import numpy as np
import py_paddle.swig_paddle as api
from paddle.trainer_config_helpers import *

import cifar_util

# some global parameter.
image_size = 32
data_size = 3 * image_size * image_size
label_size = 10


def optimizer_config():
    """Function to config optimizer."""
    settings(
        batch_size=128,
        learning_rate=0.1 / 128.0,
        learning_method=MomentumOptimizer(0.9),
        regularization=L2Regularization(0.0005 * 128))


def network_config():
    """Function to config neural network."""
    img = data_layer(name='image', size=data_size)
    lbl = data_layer(name='label', size=label_size)
    hidden1 = fc_layer(input=img, size=200)
    hidden2 = fc_layer(input=hidden1, size=200)
    inference = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())
    cost = classification_cost(input=inference, label=lbl)
    outputs(cost)


def init_parameter(gradient_machine):
    """Function to init parameter inside gradient machine"""
    assert isinstance(gradient_machine, api.GradientMachine)
    for each_param in gradient_machine.getParameters():
        assert isinstance(each_param, api.Parameter)
        array_size = len(each_param)
        array = np.random.uniform(-1.0, 1.0, array_size).astype('float32')
        each_param.getBuf(api.PARAMETER_VALUE).copyFromNumpyArray(array)


def main():
    api.initPaddle("-use_gpu=false", "-trainer_count=4")  # use 4 cpu cores

    # prepare cifar-10 data.
    cifar_data = cifar_util.Cifar10Data(
        img_size=image_size,
        mean_img_size=image_size,
        num_classes=label_size,
        batch_size=128,
        train_file_list='data/cifar-out/batches/train.txt',
        test_file_list='data/cifar-out/batches/test.txt',
        meta='data/cifar-out/batches/batches.meta')

    # get enable_types for each optimizer.
    # enable_types = [value, gradient, momentum, etc]
    # For each optimizer(SGD, Adam), GradientMachine should enable different
    # buffers.
    opt_config_proto = parse_optimizer_config(optimizer_config)
    opt_config = api.OptimizationConfig.createFromProto(opt_config_proto)
    _temp_optimizer_ = api.ParameterOptimizer.create(opt_config)
    enable_types = _temp_optimizer_.getParameterTypes()

    # Create Simple Gradient Machine.
    model_config = parse_network_config(network_config)
    gradient_machine = api.GradientMachine.createFromConfigProto(
        model_config, api.CREATE_MODE_NORMAL, enable_types)

    # This type check is not useful. Only enable type hint in IDE.
    # Such as PyCharm
    assert isinstance(gradient_machine, api.GradientMachine)

    # Initialize Parameter by numpy.
    init_parameter(gradient_machine=gradient_machine)

    # Create Local Updater. Local means not run in cluster.
    # For a cluster training, here we can change to createRemoteUpdater
    # in future.
    updater = api.ParameterUpdater.createLocalUpdater(opt_config)
    assert isinstance(updater, api.ParameterUpdater)

    # Initialize ParameterUpdater.
    updater.init(gradient_machine)

    # start gradient machine.
    # the gradient machine must be started before invoke forward/backward.
    # not just for training, but also for inference.
    gradient_machine.start()

    # evaluator can print error rate, etc. It is a C++ class.
    batch_evaluator = gradient_machine.makeEvaluator()
    test_evaluator = gradient_machine.makeEvaluator()

    # output_arguments is Neural Network forward result. Here is not useful, just passed
    # to gradient_machine.forward
    output_arguments = api.Arguments.createArguments(0)

    for pass_id in xrange(3):  # we train 2 passes.
        updater.startPass()

        for batch_id, data_batch in enumerate(cifar_data.train_data()()):
            # data_batch is input images.
            # here, for online learning, we could get data_batch from network.

            # Start update one batch.
            pass_type = updater.startBatch(len(data_batch))

            # Start BatchEvaluator.
            # batch_evaluator can be used between start/finish.
            batch_evaluator.start()

            # forwardBackward is a shortcut for forward and backward.
            # It is sometimes faster than invoke forward/backward separately,
            # because in GradientMachine, it may be async.
            gradient_machine.forwardBackward(
                cifar_data.data_converter.convert(data_batch), output_arguments,
                pass_type)

            for each_param in gradient_machine.getParameters():
                updater.update(each_param)

            # Get cost. We use numpy to calculate total cost for this batch.
            cost_vec = output_arguments.getSlotValue(0)
            cost_vec = cost_vec.copyToNumpyMat()
            cost = cost_vec.sum() / len(data_batch)

            # Make evaluator works.
            gradient_machine.eval(batch_evaluator)

            # Print logs.
            print 'Pass id', pass_id, 'Batch id', batch_id, 'with cost=', \
                cost, batch_evaluator

            batch_evaluator.finish()
            # Finish batch.
            #  * will clear gradient.
            #  * ensure all values should be updated.
            updater.finishBatch(cost)

        # testing stage. use test data set to test current network.
        updater.apply()
        test_evaluator.start()
        for data_batch in cifar_data.test_data():
            # in testing stage, only forward is needed.
            gradient_machine.forward(
                cifar_data.data_converter.convert(data_batch), output_arguments,
                api.PASS_TEST)
            gradient_machine.eval(test_evaluator)

        # print error rate for test data set
        print 'Pass', pass_id, ' test evaluator: ', test_evaluator
        test_evaluator.finish()
        updater.restore()

        updater.catchUpWith()
        params = gradient_machine.getParameters()
        for each_param in params:
            assert isinstance(each_param, api.Parameter)
            value = each_param.getBuf(api.PARAMETER_VALUE)
            value = value.copyToNumpyArray()

            # Here, we could save parameter to every where you want
            print each_param.getName(), value

        updater.finishPass()

    gradient_machine.finish()


if __name__ == '__main__':
    main()
