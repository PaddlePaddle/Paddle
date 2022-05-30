# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set


class TrtConvertMultiHeadMatmulTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(batch, dim1):
            return np.random.random((batch, dim1, 768)).astype(np.float32)

        def generate_input2(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_weight1():
            return np.random.random((768, 768)).astype(np.float32)

        def generate_weight2():
            return np.random.random(768).astype(np.float32)

        for batch in [1, 2, 4]:
            self.batch = batch
            for reshape_shape in [[0, 0, 12, 64]]:
                for dim1 in [128]:
                    input2_shapes = [[batch, reshape_shape[2], dim1, dim1],
                                     [batch, 1, 1, dim1]]
                    for input2_shape in input2_shapes:
                        for axis in [0]:
                            dics = [{
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1
                            }, {
                                "axis": 2
                            }, {
                                "shape": reshape_shape
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1
                            }, {
                                "axis": 2
                            }, {
                                "shape": reshape_shape
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1
                            }, {
                                "axis": 2
                            }, {
                                "shape": reshape_shape
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "scale": 0.125,
                                "bias": 0.0,
                                "bias_after_scale": True
                            }, {
                                "alpha": 1.0,
                                "transpose_X": False,
                                "transpose_Y": True,
                                "fused_reshape_X": [],
                                "fused_reshape_Y": [],
                                "fused_transpose_X": [],
                                "fused_transpose_Y": [],
                                "fused_reshape_Out": [],
                                "fused_transpose_Out": []
                            }, {
                                "axis": axis
                            }, {
                                "axis": -1,
                                "is_test": True
                            }, {
                                "seed": 0,
                                "dropout_prob": 0.10000000149011612,
                                "dropout_implementation": "upscale_in_train",
                                "fix_seed": False,
                                "is_test": True
                            }, {
                                "alpha": 1.0,
                                "transpose_X": False,
                                "transpose_Y": False,
                                "fused_reshape_X": [],
                                "fused_reshape_Y": [],
                                "fused_transpose_X": [],
                                "fused_transpose_Y": [],
                                "fused_reshape_Out": [],
                                "fused_transpose_Out": []
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "shape": [0, 0, 768]
                            }, {
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1
                            }]

                            ops_config = [
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul1_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul1_output"]
                                    },
                                    "op_attrs": dics[0]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul1_output"],
                                        "Y": ["elementwise_add1_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add1_output"]
                                    },
                                    "op_attrs": dics[1]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add1_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape21_output"],
                                        "XShape": ["reshape21_output_xshape"]
                                    },
                                    "op_attrs": dics[2]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["reshape21_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose21_output"],
                                        "XShape":
                                        ["transpose21_output_xshape"]
                                    },
                                    "op_attrs": dics[3]
                                },
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul2_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul2_output"]
                                    },
                                    "op_attrs": dics[4]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul2_output"],
                                        "Y": ["elementwise_add2_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add2_output"]
                                    },
                                    "op_attrs": dics[5]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add2_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape22_output"],
                                        "XShape": ["reshape22_output_xshape"]
                                    },
                                    "op_attrs": dics[6]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["reshape22_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose22_output"],
                                        "XShape":
                                        ["transpose22_output_xshape"]
                                    },
                                    "op_attrs": dics[7]
                                },
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul3_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul3_output"]
                                    },
                                    "op_attrs": dics[8]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul3_output"],
                                        "Y": ["elementwise_add3_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add3_output"]
                                    },
                                    "op_attrs": dics[9]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add3_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape23_output"],
                                        "XShape": ["reshape23_output_xshape"]
                                    },
                                    "op_attrs": dics[10]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["reshape23_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose23_output"],
                                        "XShape":
                                        ["transpose23_output_xshape"]
                                    },
                                    "op_attrs": dics[11]
                                },
                                {
                                    "op_type": "scale",
                                    "op_inputs": {
                                        "X": ["transpose23_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["scale_output"]
                                    },
                                    "op_attrs": dics[12]
                                },
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["scale_output"],
                                        "Y": ["transpose22_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["matmul1_output"]
                                    },
                                    "op_attrs": dics[13]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["matmul1_output"],
                                        "Y": ["input_data2"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add4_output"]
                                    },
                                    "op_attrs": dics[14]
                                },
                                {
                                    "op_type": "softmax",
                                    "op_inputs": {
                                        "X": ["elementwise_add4_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["softmax_output"]
                                    },
                                    "op_attrs": dics[15]
                                },
                                {
                                    "op_type": "dropout",
                                    "op_inputs": {
                                        "X": ["softmax_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["dropout3_output"]
                                    },
                                    "op_attrs": dics[16]
                                },
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["dropout3_output"],
                                        "Y": ["transpose21_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["matmul2_output"]
                                    },
                                    "op_attrs": dics[17]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["matmul2_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose24_output"],
                                        "XShape":
                                        ["transpose24_output_xshape"]
                                    },
                                    "op_attrs": dics[18]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["transpose24_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape24_output"],
                                        "XShape": ["reshape24_output_xshape"]
                                    },
                                    "op_attrs": dics[19]
                                },
                                # In order to fuse ops with 
                                # multihead_matmul_fuse_pass_v2, the last op
                                # must be mul.
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["reshape24_output"],
                                        "Y": ["mul4_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul4_output"]
                                    },
                                    "op_attrs": dics[20]
                                }
                            ]
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={
                                    "mul1_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "mul2_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "mul3_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "mul4_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "elementwise_add1_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)),
                                    "elementwise_add2_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)),
                                    "elementwise_add3_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)),
                                },
                                inputs={
                                    "input_data1": TensorConfig(
                                        data_gen=partial(generate_input1, batch,
                                                         dim1)),
                                    "input_data2": TensorConfig(
                                        data_gen=partial(generate_input2,
                                                         input2_shape)),
                                },
                                outputs=["mul4_output"])

                            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            # The last dim of input1 and input2 should be static.
            self.dynamic_shape.min_input_shape = {
                "input_data1": [1, 8, 768],
                "input_data2": [1, 1, 1, 128],
                "reshape24_output": [1, 128, 768]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data1": [16, 512, 768],
                "input_data2": [16, 256, 512, 128],
                "reshape24_output": [1, 128, 768]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data1": [8, 128, 768],
                "input_data2": [8, 32, 64, 128],
                "reshape24_output": [1, 128, 768]
            }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 4), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 4), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-4)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), (1, 3), (1e-5, 1e-5)

    def add_skip_trt_case(self):
        def teller1(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Half:
                return True
            return False

        self.add_skip_case(
            teller1, SkipReasons.TRT_NOT_IMPLEMENTED,
            "The output has diff between gpu and trt in fp16 mode.")

        def teller2(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Float32 and len(
                    self.dynamic_shape.min_input_shape) != 0 and self.batch > 2:
                return True
            return False

        self.add_skip_case(
            teller2, SkipReasons.TRT_NOT_IMPLEMENTED,
            "The output has diff between gpu and trt when dynamic fp32 mode and batch size > 2."
        )

        def teller3(program_config, predictor_config):
            if self.trt_param.precision == paddle_infer.PrecisionType.Int8:
                return True
            return False

        self.add_skip_case(
            teller3, SkipReasons.TRT_NOT_IMPLEMENTED,
            "The output has diff between gpu and trt in int8 mode.")

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


class TrtConvertMultiHeadMatmulTestInt8(TrtConvertMultiHeadMatmulTest):
    def sample_program_configs(self):
        def generate_input1(batch, dim1):
            return np.random.random((batch, dim1, 768)).astype(np.float32)

        def generate_input2(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_weight1():
            return np.random.random((768, 768)).astype(np.float32)

        def generate_weight2():
            return np.random.random(768).astype(np.float32)

        for batch in [1, 2, 4]:
            self.batch = batch
            for reshape_shape in [[0, 0, 12, 64]]:
                for dim1 in [128]:
                    input2_shapes = [[batch, reshape_shape[2], dim1, dim1],
                                     [batch, 1, 1, dim1]]
                    for input2_shape in input2_shapes:
                        for axis in [0]:
                            dics = [{
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1,
                                "enable_int8": True,
                                "Input_scale": 1.0,
                            }, {
                                "axis": 2,
                                "out_threshold": 1.0,
                            }, {
                                "shape": reshape_shape
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1,
                                "enable_int8": True,
                                "Input_scale": 1.0,
                            }, {
                                "axis": 2,
                                "out_threshold": 1.0,
                            }, {
                                "shape": reshape_shape
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1,
                                "enable_int8": True,
                                "Input_scale": 1.0,
                            }, {
                                "axis": 2,
                                "out_threshold": 1.0,
                            }, {
                                "shape": reshape_shape
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "scale": 0.125,
                                "bias": 0.0,
                                "bias_after_scale": True
                            }, {
                                "alpha": 1.0,
                                "transpose_X": False,
                                "transpose_Y": True,
                                "fused_reshape_X": [],
                                "fused_reshape_Y": [],
                                "fused_transpose_X": [],
                                "fused_transpose_Y": [],
                                "fused_reshape_Out": [],
                                "fused_transpose_Out": []
                            }, {
                                "axis": axis
                            }, {
                                "axis": -1,
                                "is_test": True
                            }, {
                                "seed": 0,
                                "dropout_prob": 0.10000000149011612,
                                "dropout_implementation": "upscale_in_train",
                                "fix_seed": False,
                                "is_test": True
                            }, {
                                "alpha": 1.0,
                                "transpose_X": False,
                                "transpose_Y": False,
                                "fused_reshape_X": [],
                                "fused_reshape_Y": [],
                                "fused_transpose_X": [],
                                "fused_transpose_Y": [],
                                "fused_reshape_Out": [],
                                "fused_transpose_Out": []
                            }, {
                                "axis": [0, 2, 1, 3]
                            }, {
                                "shape": [0, 0, 768]
                            }, {
                                "x_num_col_dims": 2,
                                "y_num_col_dims": 1
                            }]

                            ops_config = [
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul1_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul1_output"]
                                    },
                                    "op_attrs": dics[0]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul1_output"],
                                        "Y": ["elementwise_add1_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add1_output"]
                                    },
                                    "op_attrs": dics[1]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add1_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape21_output"],
                                        "XShape": ["reshape21_output_xshape"]
                                    },
                                    "op_attrs": dics[2]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["reshape21_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose21_output"],
                                        "XShape":
                                        ["transpose21_output_xshape"]
                                    },
                                    "op_attrs": dics[3]
                                },
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul2_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul2_output"]
                                    },
                                    "op_attrs": dics[4]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul2_output"],
                                        "Y": ["elementwise_add2_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add2_output"]
                                    },
                                    "op_attrs": dics[5]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add2_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape22_output"],
                                        "XShape": ["reshape22_output_xshape"]
                                    },
                                    "op_attrs": dics[6]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["reshape22_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose22_output"],
                                        "XShape":
                                        ["transpose22_output_xshape"]
                                    },
                                    "op_attrs": dics[7]
                                },
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["input_data1"],
                                        "Y": ["mul3_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul3_output"]
                                    },
                                    "op_attrs": dics[8]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["mul3_output"],
                                        "Y": ["elementwise_add3_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add3_output"]
                                    },
                                    "op_attrs": dics[9]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["elementwise_add3_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape23_output"],
                                        "XShape": ["reshape23_output_xshape"]
                                    },
                                    "op_attrs": dics[10]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["reshape23_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose23_output"],
                                        "XShape":
                                        ["transpose23_output_xshape"]
                                    },
                                    "op_attrs": dics[11]
                                },
                                {
                                    "op_type": "scale",
                                    "op_inputs": {
                                        "X": ["transpose23_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["scale_output"]
                                    },
                                    "op_attrs": dics[12]
                                },
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["scale_output"],
                                        "Y": ["transpose22_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["matmul1_output"]
                                    },
                                    "op_attrs": dics[13]
                                },
                                {
                                    "op_type": "elementwise_add",
                                    "op_inputs": {
                                        "X": ["matmul1_output"],
                                        "Y": ["input_data2"]
                                    },
                                    "op_outputs": {
                                        "Out": ["elementwise_add4_output"]
                                    },
                                    "op_attrs": dics[14]
                                },
                                {
                                    "op_type": "softmax",
                                    "op_inputs": {
                                        "X": ["elementwise_add4_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["softmax_output"]
                                    },
                                    "op_attrs": dics[15]
                                },
                                {
                                    "op_type": "dropout",
                                    "op_inputs": {
                                        "X": ["softmax_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["dropout3_output"]
                                    },
                                    "op_attrs": dics[16]
                                },
                                {
                                    "op_type": "matmul",
                                    "op_inputs": {
                                        "X": ["dropout3_output"],
                                        "Y": ["transpose21_output"],
                                    },
                                    "op_outputs": {
                                        "Out": ["matmul2_output"]
                                    },
                                    "op_attrs": dics[17]
                                },
                                {
                                    "op_type": "transpose2",
                                    "op_inputs": {
                                        "X": ["matmul2_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["transpose24_output"],
                                        "XShape":
                                        ["transpose24_output_xshape"]
                                    },
                                    "op_attrs": dics[18]
                                },
                                {
                                    "op_type": "reshape2",
                                    "op_inputs": {
                                        "X": ["transpose24_output"]
                                    },
                                    "op_outputs": {
                                        "Out": ["reshape24_output"],
                                        "XShape": ["reshape24_output_xshape"]
                                    },
                                    "op_attrs": dics[19]
                                },
                                # In order to fuse ops with 
                                # multihead_matmul_fuse_pass_v2, the last op
                                # must be mul.
                                {
                                    "op_type": "mul",
                                    "op_inputs": {
                                        "X": ["reshape24_output"],
                                        "Y": ["mul4_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["mul4_output"]
                                    },
                                    "op_attrs": dics[20]
                                }
                            ]
                            ops = self.generate_op_config(ops_config)

                            program_config = ProgramConfig(
                                ops=ops,
                                weights={
                                    "mul1_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "mul2_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "mul3_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "mul4_weight": TensorConfig(
                                        data_gen=partial(generate_weight1)),
                                    "elementwise_add1_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)),
                                    "elementwise_add2_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)),
                                    "elementwise_add3_weight": TensorConfig(
                                        data_gen=partial(generate_weight2)),
                                },
                                inputs={
                                    "input_data1": TensorConfig(
                                        data_gen=partial(generate_input1, batch,
                                                         dim1)),
                                    "input_data2": TensorConfig(
                                        data_gen=partial(generate_input2,
                                                         input2_shape)),
                                },
                                outputs=["mul4_output"])

                            yield program_config


if __name__ == "__main__":
    unittest.main()
