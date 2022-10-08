#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid.core as core
from paddle.static import program_guard, Program
from test_randperm_op import check_randperm_out, error_msg, convert_dtype

paddle.enable_static()


class TestRandpermOp(OpTest):
    """ Test randperm op."""

    def setUp(self):
        self.set_npu()
        self.op_type = "randperm"
        self.n = 200
        self.dtype = "int64"

        self.inputs = {}
        self.outputs = {"Out": np.zeros((self.n)).astype(self.dtype)}
        self.init_attrs()
        self.attrs = {
            "n": self.n,
            "dtype": convert_dtype(self.dtype),
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def _get_places(self):
        return [paddle.NPUPlace(0)]

    def init_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        self.assertTrue(check_randperm_out(self.n, out_np),
                        msg=error_msg(out_np))


class TestRandpermOpN(TestRandpermOp):

    def init_attrs(self):
        self.n = 10000


class TestRandpermOpInt32(TestRandpermOp):

    def init_attrs(self):
        self.dtype = "int32"


class TestRandpermOpFloat32(TestRandpermOp):

    def init_attrs(self):
        self.dtype = "float32"


class TestRandpermOpFloat64(TestRandpermOp):

    def init_attrs(self):
        self.dtype = "float64"


class TestRandpermOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            self.assertRaises(ValueError, paddle.randperm, -3)
            self.assertRaises(TypeError, paddle.randperm, 10, 'int8')


class TestRandpermAPI(unittest.TestCase):

    def test_out(self):
        n = 10
        place = paddle.NPUPlace(0)
        with program_guard(Program(), Program()):
            x1 = paddle.randperm(n)
            x2 = paddle.randperm(n, 'float32')

            exe = paddle.static.Executor(place)
            res = exe.run(fetch_list=[x1, x2])

            self.assertEqual(res[0].dtype, np.int64)
            self.assertEqual(res[1].dtype, np.float32)
            self.assertTrue(check_randperm_out(n, res[0]))
            self.assertTrue(check_randperm_out(n, res[1]))


class TestRandpermImperative(unittest.TestCase):

    def test_out(self):
        paddle.disable_static(paddle.NPUPlace(0))
        n = 10
        for dtype in ['int32', np.int64, 'float32', 'float64']:
            data_p = paddle.randperm(n, dtype)
            data_np = data_p.numpy()
            self.assertTrue(check_randperm_out(n, data_np),
                            msg=error_msg(data_np))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
