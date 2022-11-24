# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
=======
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
<<<<<<< HEAD
from typing import Optional, List, Callable, Dict, Any, Set
=======
from typing import Any, Dict, List
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
import unittest


class TrtConvertSkipLayernormTest(TrtLayerAutoScanTest):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        outputs = program_config.outputs

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

<<<<<<< HEAD
        #The input dimension should be less than or equal to the set axis.
=======
        # The input dimension should be less than or equal to the set axis.
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        if 'begin_norm_axis' in attrs[0] and attrs[0]['begin_norm_axis'] >= 0:
            if len(inputs['inputX_data'].shape) <= attrs[0]['begin_norm_axis']:
                return False
        return True

    def sample_program_configs(self):
<<<<<<< HEAD

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        def generate_input1(attrs: List[Dict[str, Any]], batch):
            return np.ones([batch, 128, 768]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            return np.ones([batch, 128, 768]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([768]).astype(np.float32)

        def generate_weight2(attrs: List[Dict[str, Any]]):
            return np.random.random([768]).astype(np.float32)

        for batch in [4]:
            for epsilon in [1e-5]:
                for begin_norm_axis in [2]:
                    for enable_int8 in [False, True]:
<<<<<<< HEAD
                        dics = [{
                            "epsilon": epsilon,
                            "begin_norm_axis": begin_norm_axis,
                        }, {}]

                        ops_config = [{
                            "op_type": "elementwise_add",
                            "op_inputs": {
                                "X": ["inputX_data"],
                                "Y": ["EleBias"]
                            },
                            "op_outputs": {
                                "Out": ["bias_out"]
                            },
                            "op_attrs": {
                                "axis": -1
                            }
                        }, {
                            "op_type": "elementwise_add",
                            "op_inputs": {
                                "X": ["bias_out"],
                                "Y": ["inputY_data"]
                            },
                            "op_outputs": {
                                "Out": ["ele_out"]
                            },
                            "op_attrs": {
                                "axis": -1
                            }
                        }, {
                            "op_type": "layer_norm",
                            "op_inputs": {
                                "X": ["ele_out"],
                                "Bias": ["Bias"],
                                "Scale": ["Scale"]
                            },
                            "op_outputs": {
                                "Y": ["layernorm_out"],
                                "Mean": ["Mean"],
                                "Variance": ["Variance"]
                            },
                            "op_attrs": dics[0]
                        }]
=======
                        dics = [
                            {
                                "epsilon": epsilon,
                                "begin_norm_axis": begin_norm_axis,
                            },
                            {},
                        ]

                        ops_config = [
                            {
                                "op_type": "elementwise_add",
                                "op_inputs": {
                                    "X": ["inputX_data"],
                                    "Y": ["EleBias"],
                                },
                                "op_outputs": {"Out": ["bias_out"]},
                                "op_attrs": {"axis": -1},
                            },
                            {
                                "op_type": "elementwise_add",
                                "op_inputs": {
                                    "X": ["bias_out"],
                                    "Y": ["inputY_data"],
                                },
                                "op_outputs": {"Out": ["ele_out"]},
                                "op_attrs": {"axis": -1},
                            },
                            {
                                "op_type": "layer_norm",
                                "op_inputs": {
                                    "X": ["ele_out"],
                                    "Bias": ["Bias"],
                                    "Scale": ["Scale"],
                                },
                                "op_outputs": {
                                    "Y": ["layernorm_out"],
                                    "Mean": ["Mean"],
                                    "Variance": ["Variance"],
                                },
                                "op_attrs": dics[0],
                            },
                        ]
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
<<<<<<< HEAD
                                "Bias":
                                TensorConfig(
                                    data_gen=partial(generate_weight1, dics)),
                                "Scale":
                                TensorConfig(
                                    data_gen=partial(generate_weight2, dics)),
                                "EleBias":
                                TensorConfig(
                                    data_gen=partial(generate_weight2, dics))
                            },
                            inputs={
                                "inputX_data":
                                TensorConfig(data_gen=partial(
                                    generate_input1, dics, batch)),
                                "inputY_data":
                                TensorConfig(data_gen=partial(
                                    generate_input2, dics, batch))
                            },
                            outputs=["ele_out", "layernorm_out"])
=======
                                "Bias": TensorConfig(
                                    data_gen=partial(generate_weight1, dics)
                                ),
                                "Scale": TensorConfig(
                                    data_gen=partial(generate_weight2, dics)
                                ),
                                "EleBias": TensorConfig(
                                    data_gen=partial(generate_weight2, dics)
                                ),
                            },
                            inputs={
                                "inputX_data": TensorConfig(
                                    data_gen=partial(
                                        generate_input1, dics, batch
                                    )
                                ),
                                "inputY_data": TensorConfig(
                                    data_gen=partial(
                                        generate_input2, dics, batch
                                    )
                                ),
                            },
                            outputs=["ele_out", "layernorm_out"],
                        )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

                        yield program_config

    def sample_predictor_configs(
<<<<<<< HEAD
            self, program_config) -> (paddle_infer.Config, List[int], float):

=======
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "inputX_data": [4, 128, 768],
                "inputY_data": [4, 128, 768],
                "Bias": [768],
<<<<<<< HEAD
                "Scale": [768]
=======
                "Scale": [768],
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            }
            self.dynamic_shape.max_input_shape = {
                "inputX_data": [4, 128, 768],
                "inputY_data": [4, 128, 768],
                "Bias": [768],
<<<<<<< HEAD
                "Scale": [768]
=======
                "Scale": [768],
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            }
            self.dynamic_shape.opt_input_shape = {
                "inputX_data": [4, 128, 768],
                "inputY_data": [4, 128, 768],
                "Bias": [768],
<<<<<<< HEAD
                "Scale": [768]
=======
                "Scale": [768],
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 4

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # just support dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
<<<<<<< HEAD
            attrs, True), 1e-2  # atol=1e-2 while rtol is 1e-8
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), 1e-2  # atol=1e-2 while rtol is 1e-8
=======
            attrs, True
        ), 1e-2  # atol=1e-2 while rtol is 1e-8
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2  # atol=1e-2 while rtol is 1e-8
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
