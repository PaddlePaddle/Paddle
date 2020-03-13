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

from __future__ import print_function

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import AddQuantDequantPass


class QuantDequantFusePassTRTTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 224, 224], dtype="float32")
            conv1 = fluid.layers.conv2d(
                input=data, num_filters=32, filter_size=3)
            relu1 = fluid.layers.relu(conv1)
            pool1 = fluid.layers.pool2d(
                input=relu1,
                pool_size=[3, 3],
                pool_stride=[3, 3],
                pool_padding='VALID')
            conv2 = fluid.layers.conv2d(
                input=pool1, num_filters=32, filter_size=3)
            relu2 = fluid.layers.relu(conv2)
            pool2 = fluid.layers.pool2d(
                input=relu2,
                pool_size=[3, 3],
                pool_stride=[3, 3],
                pool_padding='VALID')
            conv3 = fluid.layers.conv2d(
                input=pool2, num_filters=32, filter_size=3)
            relu3 = fluid.layers.relu(conv3)
            pool3 = fluid.layers.pool2d(
                input=relu3,
                pool_size=[3, 3],
                pool_stride=[3, 3],
                pool_padding='VALID')
            fc_out = fluid.layers.fc(input=pool3, size=10, num_flatten_dims=1)
            out = fluid.layers.softmax(input=fc_out)

        self.feeds = {
            "data": np.random.randint(
                255, size=(1, 3, 224, 224)).astype("float32") / 255
        }
        self.enable_trt = True
        self.trt_parameters = QuantDequantFusePassTRTTest.TensorRTParam(
            1 << 20, 1, 3, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

        place = fluid.CPUPlace()
        executor = fluid.Executor(place)
        executor.run(self.startup_program)

        activation_quant_type = 'moving_average_abs_max'
        weight_quant_type = 'channel_wise_abs_max'
        main_graph = IrGraph(core.Graph(self.main_program.desc), for_test=False)
        transform_pass = QuantizationTransformPass(
            scope=fluid.global_scope(),
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type)
        quantizable_op_type = ['elementwise_add', 'pool2d', 'mul', 'matmul']
        add_quant_dequant_pass = AddQuantDequantPass(
            scope=fluid.global_scope(),
            place=place,
            skip_pattern=None,
            quantizable_op_type=quantizable_op_type)
        transform_pass.apply(main_graph)
        add_quant_dequant_pass.apply(main_graph)
        freeze_pass = QuantizationFreezePass(
            scope=fluid.global_scope(),
            place=place,
            weight_quantize_type=weight_quant_type)
        freeze_pass.apply(main_graph)
        self.main_program = main_graph.to_program()

    def test_check_output(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i])


if __name__ == "__main__":
    unittest.main()
