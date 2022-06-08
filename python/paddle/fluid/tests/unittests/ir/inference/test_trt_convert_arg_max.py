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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest
from program_config import TensorConfig, ProgramConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import List


class TrtConvertArgMaxTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        input_shape = program_config.inputs["arg_max_input"].shape
        axis = program_config.ops[0].attrs["axis"]
        if axis < 0:
            axis += len(input_shape)
        if len(input_shape) <= axis or axis == 0:
            return False
        return True

    def sample_program_configs(self):

        def generate_input(rank, batch):
            dims = [batch]
            for i in range(rank - 1):
                dims.append((i + 1) * 8)
            size = np.prod(dims)
            return (np.arange(size) % 10 - 5).astype("float32").reshape(dims)

        for rank in [3, 4]:
            for batch in [1, 4]:
                for axis in [-1, 0, 1, 2, 3]:
                    for keepdims in [True, False]:
                        flatten = False
                        dtype = 2
                        ops_config = [{
                            "op_type": "arg_max",
                            "op_inputs": {
                                "X": ["arg_max_input"]
                            },
                            "op_outputs": {
                                "Out": ["arg_max_out"]
                            },
                            "op_attrs": {
                                "axis": axis,
                                "keepdims": keepdims,
                                "flatten": flatten,
                                "dtype": dtype
                            }
                        }]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(
                            ops=ops,
                            weights={},
                            inputs={
                                "arg_max_input":
                                TensorConfig(data_gen=partial(
                                    generate_input, rank, batch))
                            },
                            outputs=["arg_max_out"])
                        yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.trt_param.workspace_size = 1024000
        yield self.create_inference_config(), [1, 2], 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
