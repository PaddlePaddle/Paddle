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
from paddle.base import core
from paddle.static import amp


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestStaticAmpPromoteStats(AmpTestBase):
    def check_promote_results(
        self, use_amp, dtype, level, use_promote, expected_op_calls, debug_info
    ):
        paddle.enable_static()
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
            op_stats_list[0],
            expected_fp16_calls=expected_op_calls,
            debug_info=debug_info,
        )

        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        elif paddle.device.is_compiled_with_xpu():
            place = paddle.device.XPUPlace(0)
        else:
            raise ValueError("Only support CUDA or XPU Place.")
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
            dtype,
            level,
        )
        paddle.disable_static()

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
            debug_info="TestStaticAmpPromoteStats/test_static_amp_o1",
        )

    def test_static_amp_o2(self):
        expected_fp16_calls = {
            "conv2d": 1,
            "elementwise_add": 2,
            "relu": 0,
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
            debug_info="TestStaticAmpPromoteStats/test_static_amp_o2",
        )


if __name__ == '__main__':
    unittest.main()
