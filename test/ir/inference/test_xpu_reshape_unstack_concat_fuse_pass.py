# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestReshapeUnstackConcatFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, [
            "reshape2",
            "slice",
            "reshape2",
            "unstack",
            "concat",
            "reshape2",
            "transpose2",
            "split",
        ], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        reshape_x_shape = [4, 48, 2, 16, 4096]

        reshape_op = OpConfig(
            "reshape2",
            inputs={"X": ["reshape_x"]},
            outputs={"Out": ["reshape_out"], "XShape": ["reshape_xshape"]},
            shape=[4, -1, 48, 2, 16, 4096],
        )
        unstack_op = OpConfig(
            "unstack",
            inputs={"X": ["reshape_out"]},
            outputs={
                "Y": [
                    "unstakc_out0",
                    "unstakc_out1",
                    "unstakc_out2",
                    "unstakc_out3",
                ]
            },
            axis=0,
            num=4,
        )
        concat_op = OpConfig(
            "concat",
            inputs={
                "X": [
                    "unstakc_out0",
                    "unstakc_out1",
                    "unstakc_out2",
                    "unstakc_out3",
                ]
            },
            outputs={"Out": ["concat_out"]},
            axis=-2,
        )

        slice_0s = []
        reshape_0s = []
        slice_1s = []
        reshape_1s = []
        transposes = []
        out_names = []
        for i in range(48):
            slice_0_op = OpConfig(
                "slice",
                inputs={"Input": ["concat_out"]},
                outputs={"Out": ["slice_0_" + str(i) + "_out"]},
                starts=[i],
                ends=[i + 1],
                axes=[1],
                decrease_axis=[],
            )
            slice_0s.append(slice_0_op)

            reshape_0_op = OpConfig(
                "reshape2",
                inputs={"X": ["slice_0_" + str(i) + "_out"]},
                outputs={
                    "Out": ["reshape_0_" + str(i) + "_out"],
                    "XShape": ["reshape_0_" + str(i) + "_xshape"],
                },
                shape=[-1, 2, 64, 4, 1024],
            )
            reshape_0s.append(reshape_0_op)

            slice_1_op = OpConfig(
                "slice",
                inputs={"Input": ["reshape_0_" + str(i) + "_out"]},
                outputs={"Out": ["slice_1_" + str(i) + "_out"]},
                starts=[1],
                ends=[2],
                axes=[3],
                decrease_axis=[3],
            )
            slice_1s.append(slice_1_op)

            reshape_1_op = OpConfig(
                "reshape2",
                inputs={"X": ["slice_1_" + str(i) + "_out"]},
                outputs={
                    "Out": ["reshape_1_" + str(i) + "_out"],
                    "XShape": ["reshape_1_" + str(i) + "_xshape"],
                },
                shape=[-1, 2, 64, 16, 64],
            )
            reshape_1s.append(reshape_1_op)

            transpose_op = OpConfig(
                "transpose2",
                inputs={"X": ["reshape_1_" + str(i) + "_out"]},
                outputs={
                    "Out": ["transpose_" + str(i) + "_out"],
                    "XShape": ["transpose_" + str(i) + "_xshape"],
                },
                axis=[1, 0, 3, 2, 4],
            )
            transposes.append(transpose_op)
            out_names.append("transpose_" + str(i) + "_out")

        ops = [reshape_op, unstack_op, concat_op]
        ops.extend(slice_0s)
        ops.extend(reshape_0s)
        ops.extend(slice_1s)
        ops.extend(reshape_1s)
        ops.extend(transposes)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "reshape_x": TensorConfig(shape=reshape_x_shape),
            },
            outputs=out_names,
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=1,
            min_success_num=1,
            passes=["reshape_unstack_concat_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
