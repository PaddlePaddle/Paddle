#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import core
from paddle.static import Program, program_guard

paddle.enable_static()


def check_randperm_out(n, data_np):
    assert isinstance(
        data_np, np.ndarray
    ), "The input data_np should be np.ndarray."
    gt_sorted = np.arange(n)
    out_sorted = np.sort(data_np)
    return list(gt_sorted == out_sorted)


def error_msg(data_np):
    return (
        "The sorted ground truth and sorted out should "
        + "be equal, out = "
        + str(data_np)
    )


def convert_dtype(dtype_str):
    dtype_str_list = [np.int32, np.int64, np.float32, np.float64]
    dtype_num_list = [
        core.VarDesc.VarType.INT32,
        core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP32,
        core.VarDesc.VarType.FP64,
    ]
    assert dtype_str in dtype_str_list, (
        dtype_str + " should in " + str(dtype_str_list)
    )
    return dtype_num_list[dtype_str_list.index(dtype_str)]


class XPUTestRandpermOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "randperm"
        self.use_dynamic_create_class = False

    class TestXPURandpermOp(XPUOpTest):
        """Test randperm op."""

        def setUp(self):
            self.init_op_type()
            self.initTestCase()
            self.dtype = self.in_type
            self.use_xpu = True
            self.use_mkldnn = False
            self.inputs = {}
            self.outputs = {"Out": np.zeros(self.n).astype(self.dtype)}
            self.attrs = {
                "n": self.n,
                "dtype": convert_dtype(self.dtype),
            }

        def init_op_type(self):
            self.op_type = "randperm"
            self.use_mkldnn = False

        def initTestCase(self):
            self.n = 200

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                place = paddle.XPUPlace(0)
                self.check_output_customized(self.verify_output)

        def verify_output(self, outs):
            out_np = np.array(outs[0])
            self.assertTrue(
                check_randperm_out(self.n, out_np), msg=error_msg(out_np)
            )

    class TestXPURandpermOpN(TestXPURandpermOp):
        def initTestCase(self):
            self.n = 10000

    class TestRandpermImperative(unittest.TestCase):
        def test_out(self):
            paddle.disable_static()
            n = 10
            dtype = self.in_type
            data_p = paddle.randperm(n, dtype)
            data_np = data_p.numpy()
            self.assertTrue(
                check_randperm_out(n, data_np), msg=error_msg(data_np)
            )
            paddle.enable_static()

    class TestRandpermEager(unittest.TestCase):
        def test_out(self):
            paddle.disable_static()
            n = 10
            dtype = self.in_type
            data_p = paddle.randperm(n, dtype)
            data_np = data_p.numpy()
            self.assertTrue(
                check_randperm_out(n, data_np), msg=error_msg(data_np)
            )
            paddle.enable_static()


support_types = get_xpu_op_support_types("randperm")
for stype in support_types:
    create_test_class(globals(), XPUTestRandpermOp, stype)


class TestRandpermAPI(unittest.TestCase):
    def test_out(self):
        n = 10
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
        else:
            place = paddle.CPUPlace()
        with program_guard(Program(), Program()):
            x1 = paddle.randperm(n)
            x2 = paddle.randperm(n, 'float32')

            exe = paddle.static.Executor(place)
            res = exe.run(fetch_list=[x1, x2])

            self.assertEqual(res[0].dtype, np.int64)
            self.assertEqual(res[1].dtype, np.float32)
            self.assertTrue(check_randperm_out(n, res[0]))
            self.assertTrue(check_randperm_out(n, res[1]))


if __name__ == "__main__":
    unittest.main()
