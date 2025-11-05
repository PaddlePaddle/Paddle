# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core


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
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Bugs on XPU3, disable temporarily",
)
class TestCudaAutoCast(unittest.TestCase):
    def setUp(self):
        self._conv = paddle.nn.Conv2D(1, 1, 3, bias_attr=False)
        self._linear = paddle.nn.Linear(4, 4)

    def _run_autocast_test(self, ctx):
        with paddle.autocast(
            device_type='cuda',
            enabled=True,
            dtype=paddle.float16,
            cache_enabled=True,
        ):
            out1 = self._conv(paddle.rand(shape=[1, 1, 6, 6], dtype='float32'))
            out2 = out1 + paddle.rand(shape=out1.shape, dtype='float16')
            out3 = self._linear(out2)

        self.assertEqual(out1.dtype, paddle.float16)
        self.assertEqual(out2.dtype, paddle.float16)
        self.assertEqual(out3.dtype, paddle.float32)


if __name__ == '__main__':
    unittest.main()
