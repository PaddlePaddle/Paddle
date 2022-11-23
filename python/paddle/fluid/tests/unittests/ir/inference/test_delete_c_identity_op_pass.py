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

from auto_scan_test import PassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig
import paddle.inference as paddle_infer
import unittest
import hypothesis.strategies as st


class TestDeleteCIdentityPass(PassAutoScanTest):

    def sample_predictor_configs(self, program_config):
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=8,
            workspace_size=0,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        yield config, ['relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        n = draw(st.integers(min_value=1, max_value=2))
        relu_op = OpConfig("relu",
                           inputs={"X": ["relu_x"]},
                           outputs={"Out": ["relu_out"]})
        c_identity_op = OpConfig("c_identity",
                                 inputs={"X": ["relu_out"]},
                                 outputs={"Out": ["id_out"]})
        program_config = ProgramConfig(
            ops=[relu_op, c_identity_op],
            weights={},
            inputs={"relu_x": TensorConfig(shape=[n])},
            outputs=["id_out"])
        return program_config

    def test(self):
        self.run_and_statis(max_examples=2,
                            min_success_num=2,
                            passes=["delete_c_identity_op_pass"])


if __name__ == "__main__":
    unittest.main()
