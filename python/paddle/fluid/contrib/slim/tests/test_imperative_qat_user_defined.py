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
from paddle.optimizer import Adam
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.nn import Sequential
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from paddle.nn.quant.quant_layers import QuantizedConv2DTranspose
from paddle.fluid.log_helper import get_logger

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


class CustomQAT(nn.Layer):
    def __init__(self):
        super(CustomQAT, self).__init__()
        attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=1.0))
        self.u_param = self.create_parameter(
            shape=[1], attr=attr, dtype='float32')
        self.l_param = self.create_parameter(
            shape=[1], attr=attr, dtype='float32')
        self.alpha_param = self.create_parameter(
            shape=[1], attr=attr, dtype='float32')
        self.upper = self.create_parameter(
            shape=[1], attr=attr, dtype='float32')
        self.upper.stop_gradient = True
        self.lower = self.create_parameter(
            shape=[1], attr=attr, dtype='float32')
        self.lower.stop_gradient = True

    def forward(self, x):
        def clip(x, upper, lower):
            x = x + paddle.nn.functional.relu(lower - x)
            x = x - paddle.nn.functional.relu(x - upper)
            return x

        def phi_function(x, mi, alpha, delta):
            s = 1 / (1 - alpha)
            k = paddle.log(2 / alpha - 1) * (1 / delta)
            x = (paddle.tanh((x - mi) * k)) * s
            return x

        def dequantize(x, lower_bound, delta, interval):
            x = ((x + 1) / 2 + interval) * delta + lower_bound
            return x

        bit = 8
        bit_range = 2**bit - 1

        paddle.assign(self.upper * 0.9 + self.u_param * 0.1, self.upper)
        paddle.assign(self.lower * 0.9 + self.l_param * 0.1, self.lower)
        x = clip(x, self.upper, self.lower)
        delta = (self.upper - self.lower) / bit_range
        interval = (x - self.lower) / delta
        mi = (interval + 0.5) * delta + self.l_param
        x = phi_function(x, mi, self.alpha_param, delta)
        x = dequantize(x, self.l_param, delta, interval)
        return x


class ModelForConv2dT(nn.Layer):
    def __init__(self, num_classes=10):
        super(ModelForConv2dT, self).__init__()
        self.features = nn.Conv2DTranspose(4, 6, (3, 3))
        self.fc = Linear(input_dim=600, output_dim=num_classes)

    def forward(self, inputs):
        x = self.features(inputs)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


class ImperativeLenet(paddle.nn.Layer):
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

        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


class TestUserDefinedActPreprocess(unittest.TestCase):
    def setUp(self):
        _logger.info("test act_preprocess")
        self.imperative_qat = ImperativeQuantAware(act_preprocess_layer=PACT)

    def test_quant_aware_training(self):
        imperative_qat = self.imperative_qat
        seed = 1
        np.random.seed(seed)
        paddle.static.default_main_program().random_seed = seed
        paddle.static.default_startup_program().random_seed = seed
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
        adam = Adam(learning_rate=0.001, parameters=lenet.parameters())
        dynamic_loss_rec = []
        #for CI coverage
        conv_transpose = ModelForConv2dT()
        imperative_qat.quantize(conv_transpose)
        x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)
        conv_transpose(x_var)

        def train(model):
            adam = Adam(learning_rate=0.001, parameters=model.parameters())
            epoch_num = 1
            for epoch in range(epoch_num):
                model.train()
                for batch_id, data in enumerate(train_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = paddle.to_tensor(x_data)
                    label = paddle.to_tensor(y_data)
                    out = model(img)
                    acc = paddle.metric.accuracy(out, label, k=1)
                    loss = nn.functional.loss.cross_entropy(out, label)
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    model.clear_gradients()
                    if batch_id % 50 == 0:
                        _logger.info(
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id,
                                   avg_loss.numpy(), acc.numpy()))
                        break

        def test(model):
            model.eval()
            avg_acc = [[], []]
            for batch_id, data in enumerate(test_reader()):
                x_data = np.array([x[0].reshape(1, 28, 28)
                                   for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)

                out = model(img)
                acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
                acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
                avg_acc[0].append(acc_top1.numpy())
                avg_acc[1].append(acc_top5.numpy())
                if batch_id % 100 == 0:
                    _logger.info(
                        "Test | step {}: acc1 = {:}, acc5 = {:}".format(
                            batch_id, acc_top1.numpy(), acc_top5.numpy()))

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=512, drop_last=True)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=512)
        train(lenet)
        test(lenet)


class TestUserDefinedWeightPreprocess(TestUserDefinedActPreprocess):
    def setUp(self):
        _logger.info("test weight_preprocess")
        self.imperative_qat = ImperativeQuantAware(weight_preprocess_layer=PACT)


class TestUserDefinedActQuantize(TestUserDefinedActPreprocess):
    def setUp(self):
        _logger.info("test act_quantize")
        self.imperative_qat = ImperativeQuantAware(act_quantize_layer=CustomQAT)


class TestUserDefinedWeightQuantize(TestUserDefinedActPreprocess):
    def setUp(self):
        _logger.info("test weight_quantize")
        self.imperative_qat = ImperativeQuantAware(
            weight_quantize_layer=CustomQAT)


if __name__ == '__main__':
    unittest.main()
