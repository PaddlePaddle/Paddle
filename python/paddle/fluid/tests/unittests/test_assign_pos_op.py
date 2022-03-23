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


def assign_pos(x, _cum_count):
    cum_count = np.copy(_cum_count)
    x = x.reshape(-1)
    res = np.zeros((cum_count[-1], ), dtype=np.int64)
    for i, idx in enumerate(x):
        p = cum_count[idx]
        cum_count[idx] -= 1
        if p >= 1:
            res[p - 1] = i
    return res


def count(x, upper_num):
    res = np.zeros((upper_num, )).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res


# why defining the assert function specially?
# Becasue assign_pos_op is multithread-op, which can make the order of numbers
# in each counter(bin) is random. But the numbers set is certain in each counter(bin).
np_allclose = np.allclose


def assert_allclose(res, out, cum_count):
    c0 = 0
    for c in cum_count:
        if c == c0:
            continue
        data1 = np.copy(res[c0:c])
        data2 = np.copy(out[c0:c])
        data1.sort()
        data2.sort()
        assert np_allclose(data2, data1)
        c0 = c
    return True


def get_redefined_allclose(cum_count):
    def redefined_allclose(x, y, *args, **kwargs):
        return assert_allclose(x, y, cum_count)

    return redefined_allclose


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestAssignPosOpInt64(op_test.OpTest):
    def setUp(self):
        x = np.random.randint(0, 16, size=(100, 2)).astype("int64")
        y = count(x, 16)
        cum_count = np.cumsum(y).astype(x.dtype)
        self.op_type = "assign_pos"
        self.inputs = {
            'X': x,
            "cum_count": cum_count,
            "eff_num_len": np.array([cum_count[-1]])
        }
        self.outputs = {'Out': assign_pos(x, cum_count)}
        self.cum_count = cum_count

    def test_forward(self):
        np.allclose = get_redefined_allclose(self.cum_count)
        self.check_output_with_place(paddle.CUDAPlace(0))


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestAssignPosAPI(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randint(0, 16, size=(100, 2)).astype("int64")
        y = count(self.x, 16)
        self.cum_count = np.cumsum(y).astype(self.x.dtype)
        self.out = assign_pos(self.x, self.cum_count)
        self.place = paddle.CUDAPlace(0)

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('x', self.x.shape, dtype="int64")
            cum_count = paddle.fluid.data(
                'cum_count', self.cum_count.shape, dtype="int64")
            out = utils._assign_pos(x, cum_count)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x,
                                "cum_count": self.cum_count},
                          fetch_list=[out])
            assert_allclose(res[0], self.out, self.cum_count)

    def test_api_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        cum_count = paddle.to_tensor(self.cum_count).astype(x.dtype)

        out = utils._assign_pos(x, cum_count)
        assert_allclose(out.numpy(), self.out, self.cum_count)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
