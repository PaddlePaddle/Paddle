#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


def ref_frac(x):
    return x - np.trunc(x)


class TestFracAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.uniform(-3, 3, [2, 3]).astype('float64')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with program_guard(Program()):
            input = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = paddle.frac(input)
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_frac(self.x_np)
        self.assertTrue(np.allclose(out_ref, res))

    def test__dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.frac(x)
        out_ref = ref_frac(self.x_np)
        self.assertTrue(np.allclose(out_ref, out.numpy()))


if __name__ == '__main__':
    unittest.main()
