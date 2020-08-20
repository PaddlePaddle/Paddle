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
from op_test import OpTest
import paddle
import paddle.fluid.core as core
from paddle.static import program_guard, Program


def check_randperm_out(n, data_np):
    assert isinstance(data_np, np.ndarray), \
        "The input data_np should be np.ndarray."
    gt_sorted = np.arange(n)
    out_sorted = np.sort(data_np)
    return list(gt_sorted == out_sorted)


def error_msg(data_np):
    return "The sorted ground truth and sorted out should " + \
 "be equal, out = " + str(data_np)


def convert_dtype(dtype_str):
    dtype_str_list = ["int32", "int64", "float32", "float64"]
    dtype_num_list = [
        core.VarDesc.VarType.INT32, core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP64
    ]
    assert dtype_str in dtype_str_list, dtype_str + \
        " should in " + str(dtype_str_list)
    return dtype_num_list[dtype_str_list.index(dtype_str)]


class TestRandpermOp(OpTest):
    """ Test randperm op."""

    def setUp(self):
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

    def init_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        out_np = np.array(outs[0])
        self.assertTrue(
            check_randperm_out(self.n, out_np), msg=error_msg(out_np))


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
        place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
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
        paddle.disable_static()
        n = 10
        for dtype in ['int32', np.int64, 'float32', 'float64']:
            data_p = paddle.randperm(n, dtype)
            data_np = data_p.numpy()
            self.assertTrue(
                check_randperm_out(n, data_np), msg=error_msg(data_np))
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
