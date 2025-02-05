# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest

import numpy as np

sys.path.append("../../ir/inference")
from inference_pass_test import InferencePassTest

import paddle
from paddle import base
from paddle.base.core import PassVersionChecker


class TestMKLDNNCpuBfloat16Pass(InferencePassTest):
    def setUp(self):
        self.init_data()
        with paddle.pir_utils.OldIrGuard():
            with base.program_guard(self.main_program, self.startup_program):
                x = paddle.static.data(
                    name='x', shape=[-1, *self.shape_x], dtype=self.d_type
                )

                out = paddle.transpose(x, perm=[0, 1, 2, 3])
                out = paddle.reshape(out, [0, 0, 0, 0])

                out = paddle.static.nn.fc(out, size=1)

                self.feeds = {
                    "x": np.random.random([self.bs, *self.shape_x]).astype(
                        self.d_type
                    )
                }
                self.fetch_list = [out]

    def init_data(self):
        self.bs = 8
        self.d_type = np.float32
        self.shape_x = [12, 10, 1]
        self.shape_y = [12, 1, 64]
        self.enable_mkldnn = True
        self.enable_mkldnn_bfloat16 = True

    def test_check_output(self):
        use_gpu = False
        with paddle.pir_utils.OldIrGuard():
            self.check_output_with_option(use_gpu, flatten=True)
        self.assertTrue(PassVersionChecker.IsCompatible('cpu_bfloat16_pass'))


if __name__ == "__main__":
    unittest.main()
