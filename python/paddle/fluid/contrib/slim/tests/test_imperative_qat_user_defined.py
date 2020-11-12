#   copyright (c) 2020 paddlepaddle authors. all rights reserved.
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
import paddle.nn as nn
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
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

os.environ["CPU_NUM"] = "1"

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class PACT(nn.Layer):
    def __init__(self, init_value=20):
        super(PACT, self).__init__()
        alpha_attr = paddle.ParamAttr(
            name=self.full_name() + ".pact",
            initializer=paddle.nn.initializer.Constant(value=init_value))
        self.alpha = self.create_parameter(
            shape=[1], attr=alpha_attr, dtype='float32')

    def forward(self, x):
        out_left = paddle.nn.functional.relu(x - self.alpha)
        out_right = paddle.nn.functional.relu(-self.alpha - x)
        x = x - out_left + out_right
        return x


class ImperativeLenet(fluid.dygraph.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(ImperativeLenet, self).__init__()
        self.features = Sequential(
            Conv2D(
                num_channels=1,
                num_filters=6,
                filter_size=3,
                stride=1,
                padding=1),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2),
            Conv2D(
                num_channels=6,
                num_filters=16,
                filter_size=5,
                stride=1,
                padding=0),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2))

        self.fc = Sequential(
            Linear(
                input_dim=400, output_dim=120),
            Linear(
                input_dim=120, output_dim=84),
            Linear(
                input_dim=84, output_dim=num_classes,
                act=classifier_activation))

    def forward(self, inputs):
        x = self.features(inputs)

        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)
        return x


class TestUserDefinedActPreprocess(unittest.TestCase):
    def setUp(self):
        _logger.info("test act_preprocess")
        self.imperative_qat = ImperativeQuantAware(act_preprocess=PACT)

    def test_quant_aware_training(self):
        imperative_qat = self.imperative_qat
        seed = 1
        np.random.seed(seed)
        fluid.default_main_program().random_seed = seed
        fluid.default_startup_program().random_seed = seed
        lenet = ImperativeLenet()
        fixed_state = {}
        param_init_map = {}
        for name, param in lenet.named_parameters():
            p_shape = param.numpy().shape
            p_value = param.numpy()
            if name.endswith("bias"):
                value = np.zeros_like(p_value).astype('float32')
            else:
                value = np.random.normal(
                    loc=0.0, scale=0.01,
                    size=np.product(p_shape)).reshape(p_shape).astype('float32')
            fixed_state[name] = value
            param_init_map[param.name] = value
        lenet.set_dict(fixed_state)

        imperative_qat.quantize(lenet)
        adam = AdamOptimizer(
            learning_rate=0.001, parameter_list=lenet.parameters())
        dynamic_loss_rec = []

        def train(model):
            adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=model.parameters())
            epoch_num = 1
            for epoch in range(epoch_num):
                model.train()
                for batch_id, data in enumerate(train_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    out = model(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    model.clear_gradients()
                    if batch_id % 100 == 0:
                        _logger.info(
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id,
                                   avg_loss.numpy(), acc.numpy()))

        def test(model):
            model.eval()
            avg_acc = [[], []]
            for batch_id, data in enumerate(test_reader()):
                x_data = np.array([x[0].reshape(1, 28, 28)
                                   for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)

                out = model(img)
                acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
                acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
                avg_acc[0].append(acc_top1.numpy())
                avg_acc[1].append(acc_top5.numpy())
                if batch_id % 100 == 0:
                    _logger.info(
                        "Test | step {}: acc1 = {:}, acc5 = {:}".format(
                            batch_id, acc_top1.numpy(), acc_top5.numpy()))

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=64, drop_last=True)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=64)
        train(lenet)
        test(lenet)
        print(paddle.summary(lenet, (1, 1, 28, 28)))

        paddle.jit.save(
            layer=lenet,
            path="./dynamic_quant_user_defined/model",
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])


class TestUserDefinedWeightPreprocess(TestUserDefinedActPreprocess):
    def setUp(self):
        _logger.info("test weight_preprocess")
        self.imperative_qat = ImperativeQuantAware(weight_preprocess=PACT)


class TestUserDefinedActQuantize(TestUserDefinedActPreprocess):
    def setUp(self):
        _logger.info("test act_quantize")
        self.imperative_qat = ImperativeQuantAware(act_quantize=PACT)


class TestUserDefinedWeightQuantize(TestUserDefinedActPreprocess):
    def setUp(self):
        _logger.info("test weight_quantize")
        self.imperative_qat = ImperativeQuantAware(weight_quantize=PACT)


if __name__ == '__main__':
    unittest.main()
