#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import op_test
import numpy as np
import unittest
import paddle
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.backward import append_backward
from paddle.distributed.models.moe import utils


def count(x, upper_num):
    res = np.zeros((upper_num, )).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNumberCountOpInt64(op_test.OpTest):
    def setUp(self):
        upper_num = 16
        self.op_type = "number_count"
        x = np.random.randint(-1, upper_num, size=(1000, 2)).astype('int64')
        self.inputs = {'numbers': x}
        self.outputs = {'Out': count(x, upper_num)}
        self.attrs = {"upper_range": upper_num}

    def test_forward(self):
        self.check_output_with_place(paddle.CUDAPlace(0))


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNumberCountAPI(unittest.TestCase):
    def setUp(self):
        self.upper_num = 320
        self.x = np.random.randint(
            -1, self.upper_num, size=(6000, 200)).astype('int64')
        self.out = count(self.x, self.upper_num)
        self.place = paddle.CUDAPlace(0)

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('x', self.x.shape, dtype="int64")
            out = utils._number_count(x, self.upper_num)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x}, fetch_list=[out])
            assert np.allclose(res, self.out)

    def test_api_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = utils._number_count(x, self.upper_num)
        assert np.allclose(out.numpy(), self.out)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
