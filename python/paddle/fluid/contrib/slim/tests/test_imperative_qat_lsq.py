#   copyright (c) 2022 paddlepaddle authors. all rights reserved.
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

import os
import numpy as np
import random
import time
import tempfile
import unittest
import logging

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import (
    SGDOptimizer,
    AdamOptimizer,
    MomentumOptimizer,
)
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.nn import Sequential
from paddle.nn import ReLU, ReLU6, LeakyReLU, Sigmoid, Softmax, PReLU
from paddle.nn import Linear, Conv2D, Softmax, BatchNorm2D, MaxPool2D
from paddle.fluid.log_helper import get_logger
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn.quant.quant_layers import (
    QuantizedConv2D,
    QuantizedConv2DTranspose,
)
from paddle.fluid.framework import _test_eager_guard
from imperative_test_utils import fix_model_dict

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class ImperativeLenet(fluid.dygraph.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        conv2d_w1_attr = fluid.ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = fluid.ParamAttr(name="conv2d_w_2")
        fc_w1_attr = fluid.ParamAttr(name="fc_w_1")
        fc_w2_attr = fluid.ParamAttr(name="fc_w_2")
        fc_w3_attr = fluid.ParamAttr(name="fc_w_3")
        conv2d_b2_attr = fluid.ParamAttr(name="conv2d_b_2")
        fc_b1_attr = fluid.ParamAttr(name="fc_b_1")
        fc_b2_attr = fluid.ParamAttr(name="fc_b_2")
        fc_b3_attr = fluid.ParamAttr(name="fc_b_3")
        self.features = Sequential(
            Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=conv2d_w1_attr,
                bias_attr=False,
            ),
            BatchNorm2D(6),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                weight_attr=conv2d_w2_attr,
                bias_attr=conv2d_b2_attr,
            ),
            BatchNorm2D(16),
            PReLU(),
            MaxPool2D(kernel_size=2, stride=2),
        )

        self.fc = Sequential(
            Linear(
                in_features=400,
                out_features=120,
                weight_attr=fc_w1_attr,
                bias_attr=fc_b1_attr,
            ),
            LeakyReLU(),
            Linear(
                in_features=120,
                out_features=84,
                weight_attr=fc_w2_attr,
                bias_attr=fc_b2_attr,
            ),
            Sigmoid(),
            Linear(
                in_features=84,
                out_features=num_classes,
                weight_attr=fc_w3_attr,
                bias_attr=fc_b3_attr,
            ),
            Softmax(),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        x = paddle.flatten(x, 1, -1)
        x = self.fc(x)
        return x


class TestImperativeQatLSQ(unittest.TestCase):
    def set_vars(self):
        self.weight_quantize_type = 'channel_wise_lsq_weight'
        self.activation_quantize_type = 'lsq_act'
        self.onnx_format = False
        self.fuse_conv_bn = False

    def func_qat(self):
        self.set_vars()

        imperative_qat = ImperativeQuantAware(
            weight_quantize_type=self.weight_quantize_type,
            activation_quantize_type=self.activation_quantize_type,
            fuse_conv_bn=self.fuse_conv_bn,
        )

        seed = 100
        np.random.seed(seed)
        fluid.default_main_program().random_seed = seed
        fluid.default_startup_program().random_seed = seed
        paddle.disable_static()
        lenet = ImperativeLenet()
        lenet = fix_model_dict(lenet)
        imperative_qat.quantize(lenet)
        optimizer = MomentumOptimizer(
            learning_rate=0.1, parameter_list=lenet.parameters(), momentum=0.9
        )

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=64, drop_last=True
        )
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=32)
        epoch_num = 2
        for epoch in range(epoch_num):
            lenet.train()
            for batch_id, data in enumerate(train_reader()):
                x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]
                ).astype('float32')
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )

                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                out = lenet(img)
                acc = paddle.static.accuracy(out, label)
                loss = paddle.nn.functional.cross_entropy(
                    out, label, reduction='none', use_softmax=False
                )
                avg_loss = paddle.mean(loss)

                avg_loss.backward()
                optimizer.minimize(avg_loss)
                lenet.clear_gradients()

                if batch_id % 100 == 0:
                    _logger.info(
                        "Train | At epoch {} step {}: loss = {:}, acc= {:}".format(
                            epoch, batch_id, avg_loss.numpy(), acc.numpy()
                        )
                    )

            lenet.eval()
            eval_acc_top1_list = []
            with paddle.no_grad():
                for batch_id, data in enumerate(test_reader()):

                    x_data = np.array(
                        [x[0].reshape(1, 28, 28) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape(-1, 1)
                    )
                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = lenet(img)
                    acc_top1 = paddle.static.accuracy(
                        input=out, label=label, k=1
                    )
                    acc_top5 = paddle.static.accuracy(
                        input=out, label=label, k=5
                    )

                    if batch_id % 100 == 0:
                        eval_acc_top1_list.append(float(acc_top1.numpy()))
                        _logger.info(
                            "Test | At epoch {} step {}: acc1 = {:}, acc5 = {:}".format(
                                epoch,
                                batch_id,
                                acc_top1.numpy(),
                                acc_top5.numpy(),
                            )
                        )

            # check eval acc
            eval_acc_top1 = sum(eval_acc_top1_list) / len(eval_acc_top1_list)
            print('eval_acc_top1', eval_acc_top1)
        self.assertTrue(
            eval_acc_top1 > 0.9,
            msg="The test acc {%f} is less than 0.9." % eval_acc_top1,
        )

    def test_qat(self):
        self.func_qat()


if __name__ == '__main__':
    unittest.main()
