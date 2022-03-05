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
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest


class TrtConvertPreluTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input(batch, dim1, dim2, dim3):
            shape = [batch]
            if dim1 != 0:
                shape.append(dim1)
            if dim2 != 0:
                shape.append(dim2)
            if dim3 != 0:
                shape.append(dim3)
            return np.random.random(shape).astype(np.float32)

        def generate_alpha(attrs: List[Dict[str, Any]], dim1, dim2, dim3):
            if attrs[0]["mode"] == "all":
                return np.random.random(size=(1)).astype(np.float32)
            elif attrs[0]["mode"] == "channel" and attrs[0][
                    "data_format"] == "NCHW":
                shape = [1]
                if dim1 != 0:
                    shape.append(dim1)
                if dim2 != 0:
                    shape.append(1)
                if dim3 != 0:
                    shape.append(1)
                return np.random.random(size=shape).astype(np.float32)
            elif attrs[0]["mode"] == "channel" and attrs[0][
                    "data_format"] == "NHWC":
                shape = [1]
                if dim1 != 0:
                    shape.append(1)
                if dim2 != 0:
                    shape.append(1)
                if dim3 != 0:
                    shape.append(dim3)
                return np.random.random(size=shape).astype(np.float32)
            elif attrs[0]["mode"] == "element":
                shape = [1]
                if dim1 != 0:
                    shape.append(dim1)
                if dim2 != 0:
                    shape.append(dim2)
                if dim3 != 0:
                    shape.append(dim3)
                return np.random.random(size=shape).astype(np.float32)

        for batch in [1, 4]:
            for dim1 in [0, 3]:
                for dim2 in [0, 16]:
                    for dim3 in [0, 32]:
                        self.dim1 = dim1
                        self.dim2 = dim2
                        self.dim3 = dim3

                        if dim1 == 0 and dim2 != 0:
                            continue
                        if dim1 == 0 and dim2 == 0 and dim3 != 0:
                            continue

                        for mode in ["all", "channel", "element"]:
                            for data_format in ['NCHW', 'NHWC']:
                                if mode == "channel" and dim1 == 0 and data_format == "NCHW":
                                    continue
                                if mode == "channel" and dim3 == 0 and data_format == "NHWC":
                                    continue
                                dics = [{
                                    "mode": mode,
                                    "data_format": data_format
                                }]
                                ops_config = [{
                                    "op_type": "prelu",
                                    "op_inputs": {
                                        "X": ["input_data"],
                                        "Alpha": ["alpha_weight"]
                                    },
                                    "op_outputs": {
                                        "Out": ["output_data"]
                                    },
                                    "op_attrs": dics[0]
                                }]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights={
                                        "alpha_weight": TensorConfig(
                                            data_gen=partial(generate_alpha,
                                                             dics, dim1, dim2,
                                                             dim3))
                                    },
                                    inputs={
                                        "input_data": TensorConfig(
                                            data_gen=partial(generate_input,
                                                             batch, dim1, dim2,
                                                             dim3)),
                                    },
                                    outputs=["output_data"])

                                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if self.dim1 == 0:
                self.dynamic_shape.min_input_shape = {"input_data": [1], }
                self.dynamic_shape.max_input_shape = {"input_data": [4], }
                self.dynamic_shape.opt_input_shape = {"input_data": [2], }
            else:
                if self.dim2 == 0 and self.dim3 == 0:
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 1],
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 64],
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [2, 3],
                    }
                elif self.dim2 != 0 and self.dim3 != 0:
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 1, 1, 1],
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 64, 128, 128],
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [2, 3, 16, 32],
                    }
                elif self.dim3 == 0:
                    self.dynamic_shape.min_input_shape = {
                        "input_data": [1, 1, 1],
                    }
                    self.dynamic_shape.max_input_shape = {
                        "input_data": [4, 64, 256],
                    }
                    self.dynamic_shape.opt_input_shape = {
                        "input_data": [2, 3, 128],
                    }

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if not dynamic_shape and self.dim1 == 0 and self.dim2 == 0 and self.dim3 == 0:
                return 0, 3
            return 1, 2

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5

    def add_skip_trt_case(self):
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 7000:

            def teller(program_config, predictor_config):
                if not predictor_config.tensorrt_dynamic_shape_enabled():
                    return True
                return False

            self.add_skip_case(
                teller, SkipReasons.TRT_NOT_IMPLEMENTED,
                "Need to repair the case: the output of GPU and tensorrt has diff in trt6, the prelu static plugin has bug."
            )

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
