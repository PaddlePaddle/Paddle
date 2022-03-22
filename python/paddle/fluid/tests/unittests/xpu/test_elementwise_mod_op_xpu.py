#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
from op_test_xpu import XPUOpTest
paddle.enable_static()
import random


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseModOp(XPUOpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "elementwise_mod"
        self.dtype = np.float32
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def init_input_output(self):
        self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
        self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
        self.out = np.mod(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass


################## TEST operator % ##################
class TestRemainderOp(unittest.TestCase):
    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="float32")
            y = fluid.data(name='y', shape=[2, 3], dtype="float32")

            y_1 = paddle.remainder(x, y, name='div_res')
            self.assertEqual(('div_res' in y_1.name), True)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([2, 3, 8, 7]).astype("float32")
            np_y = np.array([1, 5, 3, 3]).astype("float32")
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = paddle.remainder(x, y)
            np_z = z.numpy()
            z_expected = np.array([0, 3, 2, 1])
            # maybe assrtAllclose()
            self.assertEqual((np_z == z_expected).all(), True)

            np_x = np.array([-3.3, 11.5, -2, 3.5])
            np_y = np.array([-1.2, 2., 3.3, -2.3])
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = x % y
            z_expected = np.array([-0.9, 1.5, 1.3, -1.1])
            self.assertEqual(np.allclose(z_expected, z.numpy()), True)

            np_x = np.array([-3, 11, -2, 3])
            np_y = np.array([-1, 2, 3, -2])
            x = paddle.to_tensor(np_x, dtype="float32")
            y = paddle.to_tensor(np_y, dtype="float32")
            z = x % y
            z_expected = np.array([0, 1, 1, -1])
            self.assertEqual(np.allclose(z_expected, z.numpy()), True)


if __name__ == '__main__':
    unittest.main()
