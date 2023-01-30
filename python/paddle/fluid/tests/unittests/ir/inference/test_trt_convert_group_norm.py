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

<<<<<<< HEAD
import unittest
from functools import partial
from typing import Any, Dict, List

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer
=======
from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class TrtConvertGroupNormTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
<<<<<<< HEAD
=======
        inputs = program_config.inputs
        weights = program_config.weights
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        if attrs[0]['epsilon'] < 0 or attrs[0]['epsilon'] > 0.001:
            return False
        if attrs[0]['groups'] <= 0:
            return False
        return True

    def sample_program_configs(self):
        def generate_input(attrs: List[Dict[str, Any]], batch):
            if attrs[0]['data_layout'] == 'NCHW':
                return np.random.random([batch, 32, 64, 64]).astype(np.float32)
            else:
                return np.random.random([batch, 64, 64, 32]).astype(np.float32)

        def generate_scale():
            return np.random.randn(32).astype(np.float32)

        def generate_bias():
            return np.random.randn(32).astype(np.float32)

        for batch in [1, 2, 4]:
            for group in [1, 4, 32, -1]:
<<<<<<< HEAD
                for epsilon in [0.00001, 0.00005]:
=======
                for epsilon in [0.0001, 0.0007, -1, 1]:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    for data_layout in ['NCHW']:
                        dics = [
                            {
                                "epsilon": epsilon,
                                "groups": group,
                                "data_layout": data_layout,
<<<<<<< HEAD
                            },
                            {},
=======
                            }
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        ]
                        ops_config = [
                            {
                                "op_type": "group_norm",
                                "op_inputs": {
                                    "X": ["input_data"],
                                    "Scale": ["scale_weight"],
                                    "Bias": ["bias_weight"],
                                },
                                "op_outputs": {
                                    "Y": ["y_output"],
                                    "Mean": ["mean_output"],
                                    "Variance": ["variance_output"],
                                },
                                "op_attrs": dics[0],
                            }
                        ]
                        ops = self.generate_op_config(ops_config)

                        program_config = ProgramConfig(
                            ops=ops,
                            weights={
                                "scale_weight": TensorConfig(
                                    data_gen=partial(generate_scale)
                                ),
                                "bias_weight": TensorConfig(
                                    data_gen=partial(generate_bias)
                                ),
                            },
                            inputs={
                                "input_data": TensorConfig(
                                    data_gen=partial(
                                        generate_input, dics, batch
                                    )
                                )
                            },
                            outputs=["y_output"],
                        )

                        yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {"input_data": [1, 16, 16, 16]}
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 64, 128, 128]
            }
            self.dynamic_shape.opt_input_shape = {"input_data": [1, 32, 64, 64]}

        def clear_dynamic_shape():
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
<<<<<<< HEAD
        self.trt_param.workspace_size = 2013265920
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-2

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), 1e-5
<<<<<<< HEAD
        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.workspace_size = 2013265920

        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-2

=======
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False
        ), (1e-3, 1e-3)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), 1e-5
<<<<<<< HEAD

    def test(self):
=======
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True
        ), (1e-3, 1e-3)

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.run_test()


if __name__ == "__main__":
    unittest.main()
