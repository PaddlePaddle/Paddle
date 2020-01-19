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
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest


def maxout_forward_naive(input, groups, channel_axis):
    s0, s1, s2, s3 = input.shape
    if channel_axis == 3:
        return np.ndarray([s0, s1, s2, s3 // groups, groups], \
            buffer = input, dtype=input.dtype).max(axis=(4))
    return np.ndarray([s0, s1 // groups, groups, s2, s3], \
        buffer = input, dtype=input.dtype).max(axis=(2))


class TestMaxOutOp(OpTest):
    def setUp(self):
        self.op_type = "maxout"
        self.init_test_case()
        input = np.random.random(self.shape)
        output = self.MaxOut_forward_naive(input, self.groups, self.axis)

        self.inputs = {'X': input}
        self.attrs = {'groups': self.groups, 'axis': self.axis}

        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.MaxOut_forward_naive = maxout_forward_naive
        self.shape = [100, 6, 2, 2]
        self.groups = 2
        self.axis = 1


class TestMaxOutOpAxis(TestMaxOutOp):
    def init_test_case(self):
        self.MaxOut_forward_naive = maxout_forward_naive
        self.shape = [100, 2, 2, 6]  # NHWC format
        self.groups = 2
        self.axis = 3


class TestMaxOutOpAxisAPI(unittest.TestCase):
    def test_axis(self):
        data1 = fluid.data(name='data1', shape=[3, 6, 2, 2], dtype='float32')
        data2 = fluid.data(name='data2', shape=[3, 2, 2, 6], dtype='float32')
        out1 = fluid.layers.maxout(data1, groups=2, axis=1)
        out2 = fluid.layers.maxout(data2, groups=2, axis=-1)
        data1_np = np.random.random((3, 6, 2, 2)).astype("float32")
        data2_np = np.transpose(data1_np, [0, 2, 3, 1])

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        results = exe.run(fluid.default_main_program(),
                          feed={"data1": data1_np,
                                "data2": data2_np},
                          fetch_list=[out1, out2],
                          return_numpy=True)

        self.assertTrue(
            np.allclose(results[0], np.transpose(results[1], (0, 3, 1, 2))))

    def test_exception(self):
        input = fluid.data(name="input", shape=[2, 4, 6, 6], dtype="float32")

        def _attr_axis():
            out = fluid.layers.maxout(input, groups=2, axis=2)

        self.assertRaises(ValueError, _attr_axis)


if __name__ == '__main__':
    unittest.main()
