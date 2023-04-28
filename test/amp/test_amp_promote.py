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

import numpy as np
from amp_base_models import AmpTestBase, build_conv_model

import paddle
from paddle.static import amp

paddle.enable_static()


class TestAMPPromote(AmpTestBase):
    def check_promote_results(
        self, use_amp, dtype, level, use_promote, expected_op_calls
    ):
        (
            main_program,
            startup_program,
            optimizer,
            feed_vars,
            fetch_vars,
        ) = build_conv_model(use_amp, dtype, level, use_promote)
        self.assertEqual(main_program.num_blocks, 1)

        amp.debugging.collect_operator_stats(main_program)
        op_stats_list = amp.debugging._get_op_stats_list(main_program)

        self._check_op_calls(
            op_stats_list[0], expected_fp16_calls=expected_op_calls
        )

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)

        max_iters = 2
        x_fp32 = np.random.random(size=[1, 1, 6, 6]).astype("float32")
        losses_o1 = self.run_program(
            main_program,
            startup_program,
            optimizer,
            feed_vars,
            fetch_vars,
            place,
            exe,
            x_fp32,
            max_iters,
            level,
        )

    def test_static_amp_o1(self):
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 0,
            "relu": 0,
            "matmul_v2": 1,
            "softmax": 0,
            "reduce_mean": 0,
            "adamw": 0,
        }
        self.check_promote_results(
            True,
            'float16',
            'O1',
            use_promote=True,
            expected_op_calls=expected_fp16_calls,
        )

    def test_static_amp_o2(self):
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 2,
            "relu": 1,
            "matmul_v2": 1,
            "softmax": 1,
            "reduce_mean": 1,
            "adamw": 4,
        }
        self.check_promote_results(
            True,
            'float16',
            'O2',
            use_promote=True,
            expected_op_calls=expected_fp16_calls,
        )


if __name__ == '__main__':
    unittest.main()
