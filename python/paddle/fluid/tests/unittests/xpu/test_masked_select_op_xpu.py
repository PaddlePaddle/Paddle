#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import unittest
import sys

sys.path.append("..")

import paddle
import paddle.fluid as fluid
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


def np_masked_select(x, mask):
    result = np.empty(shape=(0), dtype=x.dtype)
    for ele, ma in zip(np.nditer(x), np.nditer(mask)):
        if ma:
            result = np.append(result, ele)
    return result.flatten()


class XPUTestMaskedSelectOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'masked_select'

    class TestMaskedSelectOp(XPUOpTest):

        def setUp(self):
            self.init()
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = "masked_select"
            self.__class__.no_need_check_grad = True

            x = np.random.random(self.shape).astype(self.dtype)
            mask = np.array(np.random.randint(2, size=self.shape, dtype=bool))
            out = np_masked_select(x, mask)
            self.inputs = {'X': x, 'Mask': mask}
            self.outputs = {'Y': out}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def init(self):
            self.shape = (50, 3)

    class TestMaskedSelectOp1(TestMaskedSelectOp):

        def init(self):
            self.shape = (6, 8, 9, 18)

    class TestMaskedSelectOp2(TestMaskedSelectOp):

        def init(self):
            self.shape = (168, )


support_types = get_xpu_op_support_types('masked_select')
for stype in support_types:
    create_test_class(globals(), XPUTestMaskedSelectOp, stype)


class TestMaskedSelectAPI(unittest.TestCase):

    def test_imperative_mode(self):
        paddle.disable_static(paddle.XPUPlace(0))
        shape = (88, 6, 8)
        np_x = np.random.random(shape).astype('float32')
        np_mask = np.array(np.random.randint(2, size=shape, dtype=bool))
        x = paddle.to_tensor(np_x)
        mask = paddle.to_tensor(np_mask)
        out = paddle.masked_select(x, mask)
        np_out = np_masked_select(np_x, np_mask)
        self.assertEqual(np.allclose(out.numpy(), np_out), True)
        paddle.enable_static()

    def test_static_mode(self):
        shape = [8, 9, 6]
        x = paddle.fluid.data(shape=shape, dtype='float32', name='x')
        mask = paddle.fluid.data(shape=shape, dtype='bool', name='mask')
        np_x = np.random.random(shape).astype('float32')
        np_mask = np.array(np.random.randint(2, size=shape, dtype=bool))

        out = paddle.masked_select(x, mask)
        np_out = np_masked_select(np_x, np_mask)

        exe = paddle.static.Executor(place=paddle.XPUPlace(0))

        res = exe.run(paddle.static.default_main_program(),
                      feed={
                          "x": np_x,
                          "mask": np_mask
                      },
                      fetch_list=[out])
        self.assertEqual(np.allclose(res, np_out), True)


class TestMaskedSelectError(unittest.TestCase):

    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):

            shape = [8, 9, 6]
            x = paddle.fluid.data(shape=shape, dtype='float32', name='x')
            mask = paddle.fluid.data(shape=shape, dtype='bool', name='mask')
            mask_float = paddle.fluid.data(shape=shape,
                                           dtype='float32',
                                           name='mask_float')
            np_x = np.random.random(shape).astype('float32')
            np_mask = np.array(np.random.randint(2, size=shape, dtype=bool))

            def test_x_type():
                paddle.masked_select(np_x, mask)

            self.assertRaises(TypeError, test_x_type)

            def test_mask_type():
                paddle.masked_select(x, np_mask)

            self.assertRaises(TypeError, test_mask_type)

            def test_mask_dtype():
                paddle.masked_select(x, mask_float)

            self.assertRaises(TypeError, test_mask_dtype)


if __name__ == '__main__':
    unittest.main()
