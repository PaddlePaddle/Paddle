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

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.append("../../quantization")
from imperative_test_utils import fix_model_dict, train_lenet

import paddle
from paddle import base
from paddle.framework import core, set_flags
from paddle.nn import (
    BatchNorm2D,
    Conv2D,
    Linear,
    MaxPool2D,
    Sequential,
    Softmax,
)
from paddle.nn.layer import LeakyReLU, PReLU, ReLU, Sigmoid
from paddle.quantization import ImperativeQuantAware

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    set_flags({"FLAGS_cudnn_deterministic": True})


def get_valid_warning_num(warning, w):
    num = 0
    for i in range(len(w)):
        if warning in str(w[i].message):
            num += 1
    return num


class ImperativeLenet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        conv2d_w1_attr = paddle.ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = paddle.ParamAttr(name="conv2d_w_2")
        fc_w1_attr = paddle.ParamAttr(name="fc_w_1")
        fc_w2_attr = paddle.ParamAttr(name="fc_w_2")
        fc_w3_attr = paddle.ParamAttr(name="fc_w_3")
        conv2d_b2_attr = paddle.ParamAttr(name="conv2d_b_2")
        fc_b1_attr = paddle.ParamAttr(name="fc_b_1")
        fc_b2_attr = paddle.ParamAttr(name="fc_b_2")
        fc_b3_attr = paddle.ParamAttr(name="fc_b_3")
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

        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


class TestImperativeOutScale(unittest.TestCase):
    def setUp(self):
        self.root_path = tempfile.TemporaryDirectory()
        self.param_save_path = os.path.join(
            self.root_path.name, "lenet.pdparams"
        )
        self.save_path = os.path.join(
            self.root_path.name, "lenet_dynamic_outscale_infer_model"
        )

    def tearDown(self):
        self.root_path.cleanup()

    def test_out_scale_acc(self):
        seed = 1
        lr = 0.001

        weight_quantize_type = 'abs_max'
        activation_quantize_type = 'moving_average_abs_max'
        imperative_out_scale = ImperativeQuantAware(
            weight_quantize_type=weight_quantize_type,
            activation_quantize_type=activation_quantize_type,
        )

        with base.dygraph.guard():
            np.random.seed(seed)
            paddle.seed(seed)

            lenet = ImperativeLenet()
            lenet = fix_model_dict(lenet)
            imperative_out_scale.quantize(lenet)

            reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32, drop_last=True
            )
            adam = paddle.optimizer.Adam(
                learning_rate=lr, parameters=lenet.parameters()
            )
            loss_list = train_lenet(lenet, reader, adam)
            lenet.eval()

        save_dict = lenet.state_dict()
        paddle.save(save_dict, self.param_save_path)

        for i in range(len(loss_list) - 1):
            self.assertTrue(
                loss_list[i] > loss_list[i + 1],
                msg='Failed to do the imperative qat.',
            )

        with base.dygraph.guard():
            lenet = ImperativeLenet()
            load_dict = paddle.load(self.param_save_path)
            imperative_out_scale.quantize(lenet)
            lenet.set_dict(load_dict)

            reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32, drop_last=True
            )
            adam = paddle.optimizer.Adam(
                learning_rate=lr, parameters=lenet.parameters()
            )
            loss_list = train_lenet(lenet, reader, adam)
            lenet.eval()

        imperative_out_scale.save_quantized_model(
            layer=lenet,
            path=self.save_path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32'
                )
            ],
        )

        for i in range(len(loss_list) - 1):
            self.assertTrue(
                loss_list[i] > loss_list[i + 1],
                msg='Failed to do the imperative qat.',
            )


if __name__ == '__main__':
    unittest.main()
