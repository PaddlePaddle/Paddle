#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


class XPUTestGather(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'gather'

    class TestXPUGatherOp(XPUOpTest):
        def setUp(self):
            self.op_type = "gather"
            self.place = paddle.XPUPlace(0)
            self.dtype = self.in_type

            self.init_config()
            xnp = np.random.random(self.x_shape).astype(
                self.dtype if self.dtype != np.uint16 else np.float32
            )
            if self.dtype == np.uint16:
                xnp = convert_float_to_uint16(xnp)
            self.inputs = {
                'X': xnp,
                'Index': np.array(self.index).astype(self.index_type),
            }
            self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

        def init_config(self):
            self.x_shape = (10, 20)
            self.index = [1, 3, 5]
            self.index_type = np.int32

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                self.check_output_with_place(self.place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestCase1(TestXPUGatherOp):
        def init_config(self):
            self.x_shape = 100
            self.index = [1, 3, 5]
            self.index_type = np.int32

    class TestCase2(TestXPUGatherOp):
        def init_config(self):
            self.x_shape = 100
            self.index = [1, 3, 5]
            self.index_type = np.int64

    class TestCase3(TestXPUGatherOp):
        def init_config(self):
            self.x_shape = (10, 20)
            self.index = [1, 3, 5]
            self.index_type = np.int32

    class TestCase4(TestXPUGatherOp):
        def init_config(self):
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': False}
            self.index = [1, 1]
            self.index_type = np.int32

    class TestCase5(TestXPUGatherOp):
        def init_config(self):
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': False}
            self.index = [1, 1, 3]
            self.index_type = np.int32

    class TestCase6(TestXPUGatherOp):
        def init_config(self):
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': True}
            self.index = [1, 3]
            self.index_type = np.int32

    class TestCase7(TestXPUGatherOp):
        def init_config(self):
            self.x_shape = (10, 20)
            self.attrs = {'overwrite': True}
            self.index = [1, 3]
            self.index_type = np.int64


class TestGatherOpEmpty(unittest.TestCase):
    def test_gather_empty_index(self):
        if paddle.is_compiled_with_xpu():
            paddle.set_device('xpu')
            paddle.disable_static()
            data = paddle.ones([10], dtype='int32')
            index = paddle.ones([], dtype='int32')
            out = paddle.gather(data, index)
            self.assertEqual(out.shape, index.shape)
            paddle.enable_static()


support_types = get_xpu_op_support_types('gather')
for stype in support_types:
    create_test_class(globals(), XPUTestGather, stype)

if __name__ == "__main__":
    unittest.main()
