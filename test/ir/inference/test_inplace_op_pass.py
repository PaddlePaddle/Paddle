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

import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestInplaceOpPass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        def generate_input():
            return np.random.random(x_shape).astype(np.float32)

        def generate_tmp1(val):
            return np.array([val]).astype(np.int32)

        def generate_tmp2(val):
            return np.array([val]).astype(np.int32)

        def generate_tmp3(val):
            return np.array([val]).astype(np.int32)

        def generate_shape(val):
            return np.array(val).astype(np.int32)

        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=10), min_size=4, max_size=4
            )
        )
        shape = [0, -1, x_shape[-1]]
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["scale_in"]},
            outputs={"Out": ["scale_out"]},
            scale=1.3,
            bias=0.1,
            bias_after_scale=False,
        )

        test_case = draw(
            st.sampled_from(
                ["simple_reshape", "shape_tensor1", "shape_tensor2"]
            )
        )

        if test_case == "simple_reshape":
            reshape_op = OpConfig(
                "reshape2",
                inputs={"X": ["scale_out"]},
                outputs={
                    "Out": ["reshape_out"],
                    "XShape": ["reshape_xshape_out"],
                },
                shape=shape,
            )
            ops = [scale_op, reshape_op]
            program_config = ProgramConfig(
                ops=ops,
                inputs={
                    "scale_in": TensorConfig(data_gen=partial(generate_input)),
                },
                weights={},
                outputs=["reshape_out"],
            )
            return program_config

        elif test_case == "shape_tensor1":
            shape = [-1, -1, x_shape[-1]]
            reshape_op = OpConfig(
                "reshape2",
                inputs={
                    "X": ["scale_out"],
                    "ShapeTensor": ["tmp1", "tmp2", "tmp3"],
                },
                outputs={
                    "Out": ["reshape_out"],
                    "XShape": ["reshape_xshape_out"],
                },
                shape=shape,
            )
            ops = [scale_op, reshape_op]
            program_config = ProgramConfig(
                ops=ops,
                inputs={
                    "scale_in": TensorConfig(data_gen=partial(generate_input)),
                    "tmp1": TensorConfig(
                        data_gen=partial(generate_tmp1, x_shape[0])
                    ),
                    "tmp2": TensorConfig(
                        data_gen=partial(generate_tmp2, x_shape[1] * x_shape[2])
                    ),
                    "tmp3": TensorConfig(
                        data_gen=partial(generate_tmp3, x_shape[-1])
                    ),
                },
                weights={},
                outputs=["reshape_out"],
            )
            return program_config

        else:
            shape = [0, -1, x_shape[-1]]
            reshape_op = OpConfig(
                "reshape2",
                inputs={"X": ["scale_out"], "Shape": ["shape"]},
                outputs={
                    "Out": ["reshape_out"],
                    "XShape": ["reshape_xshape_out"],
                },
                shape=shape,
            )
            ops = [scale_op, reshape_op]
            program_config = ProgramConfig(
                ops=ops,
                inputs={
                    "scale_in": TensorConfig(data_gen=partial(generate_input)),
                    "shape": TensorConfig(
                        data_gen=partial(
                            generate_shape,
                            [x_shape[0], x_shape[1] * x_shape[2], x_shape[3]],
                        )
                    ),
                },
                weights={},
                outputs=["reshape_out"],
            )
            return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['scale', 'reshape2'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=["inplace_op_var_pass"],
        )


if __name__ == "__main__":
    unittest.main()
