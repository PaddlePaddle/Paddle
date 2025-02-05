# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re
import unittest

import numpy as np
from fused_pass.pass_test import PassTest

import paddle
from paddle.base import core
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
)

paddle.enable_static()


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


is_sm8x = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 8
    and paddle.device.cuda.get_device_capability()[1] >= 0
)

is_sm90 = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] == 9
    and paddle.device.cuda.get_device_capability()[1] == 0
)

is_sm_supported = is_sm8x or is_sm90


def is_flashattn_supported():
    if (
        not core.is_compiled_with_cuda()
        or get_cuda_version() < 11040
        or not is_sm_supported
    ):
        return False
    return True


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must be 8.x or 90",
)
class TestConvertMEA2FA(PassTest):
    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        with paddle.pir_utils.IrGuard():
            main_prog = paddle.static.Program()
            start_prog = paddle.static.Program()
            with paddle.pir.core.program_guard(main_prog, start_prog):
                q = paddle.static.data(
                    name='q', shape=[2, 8, 32, 128], dtype="float16"
                )
                k = paddle.static.data(
                    name='k', shape=[2, 8, 32, 128], dtype="float16"
                )
                v = paddle.static.data(
                    name='v', shape=[2, 8, 32, 128], dtype="float16"
                )

                out, _ = memory_efficient_attention(q, k, v, training=False)
                self.pass_attr_list = [{'convert_MEA_to_FA': {}}]
                self.feeds = {
                    "q": np.random.random((2, 8, 32, 128)).astype("float16"),
                    "k": np.random.random((2, 8, 32, 128)).astype("float16"),
                    "v": np.random.random((2, 8, 32, 128)).astype("float16"),
                }
                self.fetch_list = [out]
                self.valid_op_map = {"pd_op.flash_attn": 1}
                yield [main_prog, start_prog], False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            self.check_pass_correct()

    def setUp(self):
        self.places.append(paddle.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()
