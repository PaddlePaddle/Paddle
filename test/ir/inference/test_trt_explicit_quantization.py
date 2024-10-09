# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved
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

import paddle
from paddle.base import core
from paddle.base.executor import global_scope
from paddle.base.framework import IrGraph
from paddle.inference import Config, PrecisionType, create_predictor
from paddle.static.quantization import QuantizationTransformPassV2


class TestExplicitQuantizationLayer:
    def setUp(self):
        paddle.enable_static()
        np.random.seed(1024)
        paddle.seed(1024)

    def inference(self, precision_mode):
        config = Config()
        config.set_model_buffer(
            self.serialized_program,
            len(self.serialized_program),
            self.serialized_params,
            len(self.serialized_params),
        )
        config.enable_use_gpu(256, 0, PrecisionType.Half)
        config.enable_memory_optim()
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=0,
            precision_mode=precision_mode,
            use_static=False,
            use_calib_mode=False,
        )
        if precision_mode == PrecisionType.Int8:
            config.enable_tensorrt_explicit_quantization()
        config.set_trt_dynamic_shape_info(*self.dynamic_shape_info)
        config.disable_glog_info()
        predictor = create_predictor(config)
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])
        input_tensor.reshape(self.input_data.shape)
        input_tensor.copy_from_cpu(self.input_data)
        predictor.run()
        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])
        output_data = output_tensor.copy_to_cpu()
        return output_data

    def test_model(self):
        self.build_program()
        baseline = self.inference(precision_mode=PrecisionType.Float32)
        predict = self.inference(precision_mode=PrecisionType.Int8)
        np.testing.assert_allclose(predict, baseline, rtol=1e-2, atol=1e-2)


@unittest.skipIf(
    paddle.inference.get_trt_compile_version() < (8, 5, 1),
    "Quantization axis is consistent with Paddle after TRT 8.5.2.",
)
class TestExplicitQuantizationConv2d(
    TestExplicitQuantizationLayer, unittest.TestCase
):
    def build_program(self):
        with paddle.pir_utils.OldIrGuard():
            # Define the inference program
            infer_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(infer_prog, startup_prog):
                input_data = paddle.static.data(
                    name='input', shape=[None, 1, 28, 28], dtype='float32'
                )
                conv = paddle.static.nn.conv2d(
                    input=input_data,
                    num_filters=2,
                    filter_size=3,
                    bias_attr=False,
                    padding=1,
                )

            # Insert QDQ nodes by QAT API
            place = paddle.CUDAPlace(0)
            scope = global_scope()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            graph = IrGraph(core.Graph(infer_prog.desc), for_test=True)
            transform_pass = QuantizationTransformPassV2(
                scope=scope,
                place=place,
                activation_quantize_type='moving_average_abs_max',
                weight_quantize_type='channel_wise_abs_max',
            )
            transform_pass.apply(graph)
            infer_prog = graph.to_program()

            # Manually sets the scale of tensors and weights
            input_scale = scope.find_var('input@scale').get_tensor()
            input_scale.set(np.array([1.0]).astype(np.float32), place)
            conv_weight = scope.find_var('conv2d_0.w_0').get_tensor()
            weight_scale = scope.find_var('conv2d_0.w_0@scale').get_tensor()
            weight_scale_np = np.max(
                np.abs(conv_weight), axis=(1, 2, 3)
            ).astype(np.float32)
            weight_scale.set(weight_scale_np, place)

            self.serialized_program = paddle.static.serialize_program(
                [input_data], [conv], program=infer_prog
            )
            self.serialized_params = paddle.static.serialize_persistables(
                [input_data], [conv], executor=exe, program=infer_prog
            )

            self.input_data = np.random.uniform(
                low=0.0, high=1.0, size=(2, 1, 28, 28)
            ).astype(np.float32)
            self.dynamic_shape_info = [
                {"input": (1, 1, 28, 28)},
                {"input": (4, 1, 28, 28)},
                {"input": (2, 1, 28, 28)},
            ]


@unittest.skipIf(
    paddle.inference.get_trt_compile_version() < (8, 5, 1),
    "Quantization axis is consistent with Paddle after TRT 8.5.2.",
)
class TestExplicitQuantizationMatmul(
    TestExplicitQuantizationLayer, unittest.TestCase
):
    def build_program(self):
        # Define the inference program
        with paddle.pir_utils.OldIrGuard():
            infer_prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            with paddle.static.program_guard(infer_prog, startup_prog):
                input_data = paddle.static.data(
                    name='input', shape=[-1, 128], dtype='float32'
                )
                linear = paddle.static.nn.fc(
                    x=input_data, size=10, bias_attr=False
                )

            # Insert QDQ nodes by QAT API
            place = paddle.CUDAPlace(0)
            scope = global_scope()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            graph = IrGraph(core.Graph(infer_prog.desc), for_test=True)
            transform_pass = QuantizationTransformPassV2(
                scope=scope,
                place=place,
                activation_quantize_type='moving_average_abs_max',
                weight_quantize_type='channel_wise_abs_max',
            )
            transform_pass.apply(graph)
            infer_prog = graph.to_program()

            # Manually sets the scale of tensors and weights
            input_scale = scope.find_var('input@scale').get_tensor()
            input_scale.set(np.array([1.0]).astype(np.float32), place)
            conv_weight = scope.find_var('fc_0.w_0').get_tensor()
            weight_scale = scope.find_var('fc_0.w_0@scale').get_tensor()
            weight_scale_np = np.max(np.abs(conv_weight), axis=(0)).astype(
                np.float32
            )
            weight_scale.set(weight_scale_np, place)

            self.serialized_program = paddle.static.serialize_program(
                [input_data], [linear], program=infer_prog
            )
            self.serialized_params = paddle.static.serialize_persistables(
                [input_data], [linear], executor=exe, program=infer_prog
            )

            self.input_data = np.random.uniform(
                low=0.0, high=1.0, size=(2, 128)
            ).astype(np.float32)
            self.dynamic_shape_info = [
                {"input": (1, 128)},
                {"input": (4, 128)},
                {"input": (2, 128)},
            ]


if __name__ == '__main__':
    unittest.main()
