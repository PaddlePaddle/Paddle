#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

from __future__ import print_function

import os
import numpy as np
import random
import unittest
import logging
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph.nn import Pool2D
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.log_helper import get_logger

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def StaticLenet(data, num_classes=10, classifier_activation='softmax'):
    conv2d_w1_attr = fluid.ParamAttr(name="conv2d_w_1")
    conv2d_w2_attr = fluid.ParamAttr(name="conv2d_w_2")
    fc_w1_attr = fluid.ParamAttr(name="fc_w_1")
    fc_w2_attr = fluid.ParamAttr(name="fc_w_2")
    fc_w3_attr = fluid.ParamAttr(name="fc_w_3")
    conv2d_b1_attr = fluid.ParamAttr(name="conv2d_b_1")
    conv2d_b2_attr = fluid.ParamAttr(name="conv2d_b_2")
    fc_b1_attr = fluid.ParamAttr(name="fc_b_1")
    fc_b2_attr = fluid.ParamAttr(name="fc_b_2")
    fc_b3_attr = fluid.ParamAttr(name="fc_b_3")
    conv1 = fluid.layers.conv2d(
        data,
        num_filters=6,
        filter_size=3,
        stride=1,
        padding=1,
        param_attr=conv2d_w1_attr,
        bias_attr=conv2d_b1_attr)
    pool1 = fluid.layers.pool2d(
        conv1, pool_size=2, pool_type='max', pool_stride=2)
    conv2 = fluid.layers.conv2d(
        pool1,
        num_filters=16,
        filter_size=5,
        stride=1,
        padding=0,
        param_attr=conv2d_w2_attr,
        bias_attr=conv2d_b2_attr)
    pool2 = fluid.layers.pool2d(
        conv2, pool_size=2, pool_type='max', pool_stride=2)

    fc1 = fluid.layers.fc(input=pool2,
                          size=120,
                          param_attr=fc_w1_attr,
                          bias_attr=fc_b1_attr)
    fc2 = fluid.layers.fc(input=fc1,
                          size=84,
                          param_attr=fc_w2_attr,
                          bias_attr=fc_b2_attr)
    fc3 = fluid.layers.fc(input=fc2,
                          size=num_classes,
                          act=classifier_activation,
                          param_attr=fc_w3_attr,
                          bias_attr=fc_b3_attr)

    return fc3


class ImperativeLenet(fluid.dygraph.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(ImperativeLenet, self).__init__()
        conv2d_w1_attr = fluid.ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = fluid.ParamAttr(name="conv2d_w_2")
        fc_w1_attr = fluid.ParamAttr(name="fc_w_1")
        fc_w2_attr = fluid.ParamAttr(name="fc_w_2")
        fc_w3_attr = fluid.ParamAttr(name="fc_w_3")
        conv2d_b1_attr = fluid.ParamAttr(name="conv2d_b_1")
        conv2d_b2_attr = fluid.ParamAttr(name="conv2d_b_2")
        fc_b1_attr = fluid.ParamAttr(name="fc_b_1")
        fc_b2_attr = fluid.ParamAttr(name="fc_b_2")
        fc_b3_attr = fluid.ParamAttr(name="fc_b_3")
        self.features = Sequential(
            Conv2D(
                num_channels=1,
                num_filters=6,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=conv2d_w1_attr,
                bias_attr=conv2d_b1_attr),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2),
            Conv2D(
                num_channels=6,
                num_filters=16,
                filter_size=5,
                stride=1,
                padding=0,
                param_attr=conv2d_w2_attr,
                bias_attr=conv2d_b2_attr),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2))

        self.fc = Sequential(
            Linear(
                input_dim=400,
                output_dim=120,
                param_attr=fc_w1_attr,
                bias_attr=fc_b1_attr),
            Linear(
                input_dim=120,
                output_dim=84,
                param_attr=fc_w2_attr,
                bias_attr=fc_b2_attr),
            Linear(
                input_dim=84,
                output_dim=num_classes,
                act=classifier_activation,
                param_attr=fc_w3_attr,
                bias_attr=fc_b3_attr))

    def forward(self, inputs):
        x = self.features(inputs)

        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)
        return x


class TestImperativeQat(unittest.TestCase):
    """
    QAT = quantization-aware training
    """

    def test_qat_save(self):
        imperative_qat = ImperativeQuantAware(
            weight_quantize_type='abs_max',
            activation_quantize_type='moving_average_abs_max')

        with fluid.dygraph.guard():
            lenet = ImperativeLenet()
            imperative_qat.quantize(lenet)
            adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=lenet.parameters())
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32)

            epoch_num = 1
            for epoch in range(epoch_num):
                lenet.train()
                for batch_id, data in enumerate(train_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = lenet(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    lenet.clear_gradients()
                    if batch_id % 100 == 0:
                        _logger.info(
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id,
                                   avg_loss.numpy(), acc.numpy()))

                lenet.eval()
                for batch_id, data in enumerate(test_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = lenet(img)
                    acc_top1 = fluid.layers.accuracy(
                        input=out, label=label, k=1)
                    acc_top5 = fluid.layers.accuracy(
                        input=out, label=label, k=5)

                    if batch_id % 100 == 0:
                        _logger.info(
                            "Test | At epoch {} step {}: acc1 = {:}, acc5 = {:}".
                            format(epoch, batch_id,
                                   acc_top1.numpy(), acc_top5.numpy()))

            # save weights
            model_dict = lenet.state_dict()
            fluid.save_dygraph(model_dict, "save_temp")

            # test the correctness of `save_quantized_model`
            data = next(test_reader())
            test_data = np.array([x[0].reshape(1, 28, 28)
                                  for x in data]).astype('float32')
            test_img = fluid.dygraph.to_variable(test_data)
            lenet.eval()
            before_save = lenet(test_img)

        # save inference quantized model
        path = "./mnist_infer_model"
        imperative_qat.save_quantized_model(
            dirname=path,
            model=lenet,
            input_shape=[(1, 28, 28)],
            input_dtype=['float32'],
            feed=[0],
            fetch=[0])
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=path, executor=exe))
        after_save, = exe.run(inference_program,
                              feed={feed_target_names[0]: test_data},
                              fetch_list=fetch_targets)

        self.assertTrue(
            np.allclose(after_save, before_save.numpy()),
            msg='Failed to save the inference quantized model.')


if __name__ == '__main__':
    unittest.main()
