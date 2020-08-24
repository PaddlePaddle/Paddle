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
from paddle.optimizer import AdamOptimizer
from paddle.fluid.contrib.slim.quantization import ImperativeOutScale
from paddle.fluid.dygraph import Sequential
from paddle.nn.layer import ReLU, LeakyReLU, Sigmoid
from paddle.fluid.dygraph.nn import BatchNorm, Conv2D, Linear, PRelu, Pool2D
from paddle.fluid.log_helper import get_logger

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


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
            BatchNorm(6),
            ReLU(),
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
            BatchNorm(16),
            PRelu('all'),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2))

        self.fc = Sequential(
            Linear(
                input_dim=400,
                output_dim=120,
                param_attr=fc_w1_attr,
                bias_attr=fc_b1_attr),
            LeakyReLU(),
            Linear(
                input_dim=120,
                output_dim=84,
                param_attr=fc_w2_attr,
                bias_attr=fc_b2_attr),
            Sigmoid(),
            Linear(
                input_dim=84,
                output_dim=num_classes,
                act=classifier_activation,
                param_attr=fc_w3_attr,
                bias_attr=fc_b3_attr))

    def forward(self, inputs):
        x = self.features(inputs)

        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x


class TestImperativeOutScale(unittest.TestCase):
    def test_dygraph_out_scale(self):
        reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=32, drop_last=True)
        param_init_map = {}
        seed = 1000
        lr = 0.1

        imperative_out_scale = ImperativeOutScale()

        with fluid.dygraph.guard():
            np.random.seed(seed)
            lenet = ImperativeLenet()
            fixed_state = {}
            for name, param in lenet.named_parameters():
                p_shape = param.numpy().shape
                p_value = param.numpy()
                if name.endswith("bias"):
                    value = np.zeros_like(p_value).astype('float32')
                else:
                    value = np.random.normal(
                        loc=0.0, scale=0.01, size=np.product(p_shape)).reshape(
                            p_shape).astype('float32')
                fixed_state[name] = value
                param_init_map[param.name] = value
            lenet.set_dict(fixed_state)

            imperative_out_scale.get_out_scale(lenet)
            adam = AdamOptimizer(
                learning_rate=lr, parameter_list=lenet.parameters())
            dynamic_loss_rec = []
            lenet.train()
            for batch_id, data in enumerate(reader()):
                x_data = np.array([x[0].reshape(1, 28, 28)
                                   for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)

                out = lenet(img)
                loss = paddle.nn.functional.cross_entropy(out, label)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                lenet.clear_gradients()
                dynamic_loss_rec.append(avg_loss.numpy()[0])
                if batch_id % 100 == 0:
                    _logger.info('{}: {}'.format('loss', avg_loss.numpy()))

            lenet.eval()
            imperative_out_scale.set_out_scale(lenet)
            op_object_list = (Conv2D, ReLU, PRelu, LeakyReLU, Sigmoid, Pool2D,
                              BatchNorm)
            out_scale_num = 0
            for name, layer in lenet.named_sublayers():
                if hasattr(layer, 'out_threshold'):
                    out_scale_num += 1
                    self.assertTrue(isinstance(layer, op_object_list))
            self.assertTrue(out_scale_num > 0)


if __name__ == '__main__':
    unittest.main()
