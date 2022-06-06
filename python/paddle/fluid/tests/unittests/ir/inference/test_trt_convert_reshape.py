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


class TrtConvertReshapeTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if self.dims == 1:
            if len(attrs[0]['shape']) != 1:
                return False

        #To test if the shape contains 0
        if len(attrs[0]['shape']) == 3:
            if attrs[0]['shape'][1] == 0:
                if self.dims != 3:
                    return False

        if len(attrs[0]['shape']) == 4:
            if attrs[0]['shape'][2] == 0:
                if self.dims != 4:
                    return False

        return True

    def sample_program_configs(self):

        def generate_input1(attrs: List[Dict[str, Any]]):
            if self.dims == 4:
                return np.ones([1, 2, 4, 6]).astype(np.float32)
            elif self.dims == 3:
                return np.ones([1, 8, 6]).astype(np.float32)
            elif self.dims == 2:
                return np.ones([1, 48]).astype(np.float32)
            elif self.dims == 1:
                return np.ones([48]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.array([1, 48]).astype(np.int32)

        def generate_shapeT1_data(attrs: List[Dict[str, Any]]):
            return np.array([2]).astype(np.int32)

        def generate_shapeT2_data(attrs: List[Dict[str, Any]]):
            return np.array([24]).astype(np.int32)

        for dims in [4, 3, 2, 1]:
            for num_input in [0, 1, 2, 3]:
                for shape in [[1, 6, 8], [1, 2, 4, 6], [1, 1, 0, 12], [1, 0, 6],
                              [1, -1, 12], [2, -1], [3, 16], [3, 4, 4], [48]]:
                    dics = [{
                        "shape": shape,
                    }, {}]
                    self.num_input = num_input
                    self.dims = dims
                    dics_intput = [{
                        "X": ["reshape_input"],
                        "Shape": ["shape_data"],
                        "ShapeTensor": ["shapeT1_data", "shapeT2_data"],
                    }, {
                        "X": ["reshape_input"],
                        "Shape": ["shape_data"],
                    }, {
                        "X": ["reshape_input"],
                        "ShapeTensor": ["shapeT1_data", "shapeT2_data"],
                    }, {
                        "X": ["reshape_input"]
                    }]

                    dics_weight = [{
                        "shape_data":
                        TensorConfig(data_gen=partial(generate_weight1, dics)),
                        "shapeT1_data":
                        TensorConfig(
                            data_gen=partial(generate_shapeT1_data, dics)),
                        "shapeT2_data":
                        TensorConfig(
                            data_gen=partial(generate_shapeT2_data, dics))
                    }, {
                        "shape_data":
                        TensorConfig(data_gen=partial(generate_weight1, dics))
                    }, {
                        "shapeT1_data":
                        TensorConfig(
                            data_gen=partial(generate_shapeT1_data, dics)),
                        "shapeT2_data":
                        TensorConfig(
                            data_gen=partial(generate_shapeT2_data, dics))
                    }, {}]

                    ops_config = [{
                        "op_type": "reshape",
                        "op_inputs": dics_intput[num_input],
                        "op_outputs": {
                            "Out": ["reshape_out"]
                        },
                        "op_attrs": dics[0]
                    }]
                    ops = self.generate_op_config(ops_config)
                    program_config = ProgramConfig(
                        ops=ops,
                        weights=dics_weight[num_input],
                        inputs={
                            "reshape_input":
                            TensorConfig(
                                data_gen=partial(generate_input1, dics))
                        },
                        outputs=["reshape_out"])

                    yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):

        def generate_dynamic_shape(attrs):
            if self.dims == 4:
                self.dynamic_shape.min_input_shape = {
                    "reshape_input": [1, 2, 4, 6]
                }
                self.dynamic_shape.max_input_shape = {
                    "reshape_input": [4, 2, 4, 6]
                }
                self.dynamic_shape.opt_input_shape = {
                    "reshape_input": [1, 2, 4, 6]
                }
            elif self.dims == 3:
                self.dynamic_shape.min_input_shape = {
                    "reshape_input": [1, 8, 6]
                }
                self.dynamic_shape.max_input_shape = {
                    "reshape_input": [4, 8, 6]
                }
                self.dynamic_shape.opt_input_shape = {
                    "reshape_input": [1, 8, 6]
                }
            elif self.dims == 2:
                self.dynamic_shape.min_input_shape = {"reshape_input": [1, 48]}
                self.dynamic_shape.max_input_shape = {"reshape_input": [4, 48]}
                self.dynamic_shape.opt_input_shape = {"reshape_input": [1, 48]}
            elif self.dims == 1:
                self.dynamic_shape.min_input_shape = {"reshape_input": [48]}
                self.dynamic_shape.max_input_shape = {"reshape_input": [48]}
                self.dynamic_shape.opt_input_shape = {"reshape_input": [48]}

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['shape'][0] > 1 and len(attrs[0]['shape']) > 1:
            pass
        else:
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
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-5

    def add_skip_trt_case(self):

        def teller1(program_config, predictor_config):
            if len(program_config.weights) >= 1:
                return True
            return False

        self.add_skip_case(teller1, SkipReasons.TRT_NOT_SUPPORT,
                           "INPUT ShapeTensor and Shape NOT SUPPORT")

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
