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
import paddle.fluid.layers as layers
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn.layer import ReLU, LeakyReLU, Sigmoid, Softmax, ReLU6
from paddle.nn import Linear, Conv2D, Softmax, BatchNorm
from paddle.fluid.dygraph.nn import Pool2D
from paddle.fluid.log_helper import get_logger

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

quant_skip_pattern_list = ['skip_qat', 'skip_quant']


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
        self.conv2d_0 = Conv2D(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv2d_w1_attr,
            bias_attr=conv2d_b1_attr)
        self.conv2d_0.skip_quant = True

        self.batch_norm_0 = BatchNorm(6)
        self.relu_0 = ReLU()
        self.pool2d_0 = Pool2D(pool_size=2, pool_type='max', pool_stride=2)
        self.conv2d_1 = Conv2D(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            weight_attr=conv2d_w2_attr,
            bias_attr=conv2d_b2_attr)
        self.conv2d_1.skip_quant = False

        self.batch_norm_1 = BatchNorm(16)
        self.relu6_0 = ReLU6()
        self.pool2d_1 = Pool2D(pool_size=2, pool_type='max', pool_stride=2)
        self.linear_0 = Linear(
            in_features=400,
            out_features=120,
            weight_attr=fc_w1_attr,
            bias_attr=fc_b1_attr)
        self.linear_0.skip_quant = True

        self.leaky_relu_0 = LeakyReLU()
        self.linear_1 = Linear(
            in_features=120,
            out_features=84,
            weight_attr=fc_w2_attr,
            bias_attr=fc_b2_attr)
        self.linear_1.skip_quant = False

        self.sigmoid_0 = Sigmoid()
        self.linear_2 = Linear(
            in_features=84,
            out_features=num_classes,
            weight_attr=fc_w3_attr,
            bias_attr=fc_b3_attr)
        self.linear_2.skip_quant = False
        self.softmax_0 = Softmax()

    def forward(self, inputs):
        x = self.conv2d_0(inputs)
        x = self.batch_norm_0(x)
        x = self.relu_0(x)
        x = self.pool2d_0(x)
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.relu6_0(x)
        x = self.pool2d_1(x)

        x = fluid.layers.flatten(x, 1)

        x = self.linear_0(x)
        x = self.leaky_relu_0(x)
        x = self.linear_1(x)
        x = self.sigmoid_0(x)
        x = self.linear_2(x)
        x = self.softmax_0(x)

        return x


class TestImperativeOutSclae(unittest.TestCase):
    def test_out_scale_acc(self):
        seed = 1000
        lr = 0.1

        imperative_out_scale = ImperativeQuantAware()

        np.random.seed(seed)
        reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=32, drop_last=True)
        lenet = ImperativeLenet()
        fixed_state = {}
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
        lenet.set_dict(fixed_state)
        imperative_out_scale.quantize(lenet)
        adam = AdamOptimizer(
            learning_rate=lr, parameter_list=lenet.parameters())
        dynamic_loss_rec = []
        lenet.train()
        for batch_id, data in enumerate(reader()):
            x_data = np.array([x[0].reshape(1, 28, 28)
                               for x in data]).astype('float32')
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(-1, 1)

            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)

            out = lenet(img)
            loss = fluid.layers.cross_entropy(out, label)
            avg_loss = fluid.layers.mean(loss)
            avg_loss.backward()
            adam.minimize(avg_loss)
            lenet.clear_gradients()
            dynamic_loss_rec.append(avg_loss.numpy()[0])
            if batch_id % 100 == 0:
                _logger.info('{}: {}'.format('loss', avg_loss.numpy()))

        lenet.eval()

        path = "./save_dynamic_quant_infer_model/lenet"
        save_dir = "./save_dynamic_quant_infer_model"

        imperative_out_scale.save_quantized_model(
            layer=lenet,
            path=path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])

        paddle.enable_static()

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)

        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=save_dir,
                executor=exe,
                model_filename="lenet" + INFER_MODEL_SUFFIX,
                params_filename="lenet" + INFER_PARAMS_SUFFIX))
        model_ops = inference_program.global_block().ops

        conv2d_count, mul_count = 0, 0
        for i, op in enumerate(model_ops):
            if op.type == 'conv2d':
                if conv2d_count > 0:
                    self.assertTrue(
                        'fake_quantize_dequantize' in model_ops[i - 1].type)
                else:
                    self.assertTrue(
                        'fake_quantize_dequantize' not in model_ops[i - 1].type)
                conv2d_count += 1

            if op.type == 'mul':
                if mul_count > 0:
                    self.assertTrue(
                        'fake_quantize_dequantize' in model_ops[i - 1].type)
                else:
                    self.assertTrue(
                        'fake_quantize_dequantize' not in model_ops[i - 1].type)
                mul_count += 1


if __name__ == '__main__':
    unittest.main()
