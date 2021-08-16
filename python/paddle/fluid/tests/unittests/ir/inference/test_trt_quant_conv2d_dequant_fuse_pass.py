# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import IrGraph
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.core import AnalysisConfig




        if quant:
            main_graph = IrGraph(
                core.Graph(self.main_program.desc), for_test=True)

            transform_pass = QuantizationTransformPass(
                scope=scope,
                place=place,
                activation_quantize_type=self.activation_quant_type,
                weight_quantize_type=self.weight_quant_type,
                quantizable_op_type=[
                    'conv2d', 'mul', 'depthwise_conv2d', 'conv2d_transpose'
                ])
            transform_pass.apply(main_graph)
            '''
            weight_scale_map = {
                "conv2d": "conv2d_0.w_0.scale",
                "mul": "fc_0.w_0.scale"
            }
            weight_scale_tensor = scope.var(weight_scale_map[
                self.quantized_op_type]).get_tensor()
            weight_scale = np.ones(self.channels).astype("float32")
            weight_scale_tensor.set(weight_scale, place)
            '''
            op_nodes = main_graph.all_op_nodes()
            for op_node in op_nodes:
                if op_node.name() in [self.quantized_op_type, "relu"]:
                    op_node.op()._set_attr("out_threshold", 0.5)
            '''
            with fluid.scope_guard(scope):
                executor.run(program=self.main_program,
                             feed=self.feeds,
                             fetch_list=self.fetch_list)
            '''
            freeze_pass = QuantizationFreezePass(
                scope=scope,
                place=place,
                weight_quantize_type=self.weight_quant_type)
            freeze_pass.apply(main_graph)
            self.main_program = main_graph.to_program() 

def fc_net(input, label):
    fc_out = fluid.layers.fc(input=input, size=100, act='relu')
    loss = fluid.layers.cross_entropy(input=fc_out, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss

def conv_net(img, label):
    conv_out = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        pool_type='max',
        act="relu")
    loss = fluid.layers.cross_entropy(input=conv_out, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss

class QuantDequantTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 32, 32], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.5),
                trainable=False)
            quantized_op_out = self.append_quantized_op(data, param_attr)
            relu_out = fluid.layers.relu(quantized_op_out)
        self.set_quant_pattern()

        self.feeds = {
            "data": np.random.random([1, 3, 32, 32]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = Conv2d_QuantDequantTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.fetch_list = [relu_out]

    def append_quantized_op(self, x, param_attr):
        return fluid.layers.conv2d(
            input=x,
            num_filters=3,
            filter_size=3,
            param_attr=param_attr,
            bias_attr=False,
            act=None)

    def set_quant_pattern(self):
        self.activation_quant_type = 'moving_average_abs_max'
        self.weight_quant_type = 'channel_wise_abs_max'
        self.quantized_op_type = 'conv2d'
        self.channels = 3

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=False, quant=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible(
                    'quant_conv2d_dequant_fuse_pass'))



class Fc_QuantDequantTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 32, 32], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.5),
                trainable=False)
            quantized_op_out = self.append_quantized_op(data, param_attr)
            relu_out = fluid.layers.relu(quantized_op_out)
        self.set_quant_pattern()

        self.feeds = {
            "data": np.random.random([1, 3, 32, 32]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = Fc_QuantDequantTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.fetch_list = [relu_out]

    def append_quantized_op(self, x, param_attr):
        return fluid.layers.fc(x,
                               size=100,
                               num_flatten_dims=1,
                               param_attr=param_attr,
                               bias_attr=False,
                               act=None)

    def set_quant_pattern(self):
        self.activation_quant_type = 'moving_average_abs_max'
        self.weight_quant_type = 'channel_wise_abs_max'
        self.quantized_op_type = 'mul'
        self.channels = 100

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=False, quant=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible(
                    'quant_conv2d_dequant_fuse_pass'))

if __name__ == "__main__":
    unittest.main()
