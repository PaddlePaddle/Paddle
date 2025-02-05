#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest

sys.path.append("../../legacy_test")

import numpy as np
from op_test import OpTest
from test_attribute_var import UnittestBase

import paddle
from paddle import base
from paddle.base import core, framework
from paddle.framework import in_pir_mode


class TestEyeOp(OpTest):
    def setUp(self):
        '''
        Test eye op with default shape
        '''
        self.python_api = paddle.eye
        self.op_type = "eye"
        self.prim_op_type = "comp"
        self.public_python_api = paddle.eye
        self.init_dtype()
        self.init_attrs()

        self.inputs = {}
        self.attrs = {
            'num_rows': self.num_columns,
            'num_columns': self.num_columns,
            'dtype': framework.convert_np_dtype_to_proto_type(self.dtype),
        }
        self.outputs = {
            'Out': np.eye(self.num_rows, self.num_columns, dtype=self.dtype)
        }

    def test_check_output(self):
        if self.dtype == np.complex64 or self.dtype == np.complex128:
            self.check_output(check_pir=True)
        else:
            self.check_output(check_pir=True, check_prim_pir=True)

    def init_dtype(self):
        self.dtype = np.int32

    def init_attrs(self):
        self.num_rows = 319
        self.num_columns = 319


class TestEyeOp1(OpTest):
    def setUp(self):
        '''
        Test eye op with default parameters
        '''
        self.python_api = paddle.eye
        self.op_type = "eye"
        self.prim_op_type = "comp"
        self.public_python_api = paddle.eye

        self.inputs = {}
        self.attrs = {'num_rows': 50}
        self.outputs = {'Out': np.eye(50, dtype=float)}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestEyeOp2(OpTest):
    def setUp(self):
        '''
        Test eye op with specified shape
        '''
        self.python_api = paddle.eye
        self.op_type = "eye"
        self.prim_op_type = "comp"
        self.public_python_api = paddle.eye

        self.inputs = {}
        self.attrs = {'num_rows': 99, 'num_columns': 1}
        self.outputs = {'Out': np.eye(99, 1, dtype=float)}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class TestEyeOp3(OpTest):
    def setUp(self):
        '''
        Test eye op with np.int32 scalar
        '''
        self.python_api = paddle.eye
        self.op_type = "eye"
        self.prim_op_type = "comp"
        self.public_python_api = paddle.eye

        self.inputs = {}
        self.attrs = {'num_rows': np.int32(99), 'num_columns': np.int32(1)}
        self.outputs = {'Out': np.eye(99, 1, dtype=float)}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)


class API_TestTensorEye(unittest.TestCase):

    def test_static_out(self):
        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10)
            place = base.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, num_columns=7, dtype="float64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[data])
            expected_result = np.eye(10, 7, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, dtype="int64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)

    def test_dynamic_out(self):
        paddle.disable_static()
        out = paddle.eye(10, dtype="int64")
        expected_result = np.eye(10, dtype="int64")
        paddle.enable_static()
        self.assertEqual((out.numpy() == expected_result).all(), True)

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):

            def test_num_rows_type_check():
                paddle.eye(-1, dtype="int64")

            self.assertRaises(TypeError, test_num_rows_type_check)

            def test_num_columns_type_check():
                paddle.eye(10, num_columns=5.2, dtype="int64")

            self.assertRaises(TypeError, test_num_columns_type_check)

            def test_num_columns_type_check1():
                paddle.eye(10, num_columns=10, dtype="int8")

            self.assertRaises(TypeError, test_num_columns_type_check1)


class TestEyeRowsCol(UnittestBase):
    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, self.path_prefix())

    def test_static(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]

            tmp = self.call_func(feat)
            out = feat + tmp

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            if not in_pir_mode():
                self.assertTrue(self.var_prefix() in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[tmp, out])
            gt = np.eye(3, 10)
            np.testing.assert_allclose(res[0], gt)
            paddle.static.save_inference_model(
                self.save_path, [x], [tmp, out], exe
            )
            # Test for Inference Predictor
            infer_outs = self.infer_prog()
            np.testing.assert_allclose(infer_outs[0], gt)

    def path_prefix(self):
        return 'eye_rows_cols'

    def var_prefix(self):
        return "Var["

    def call_func(self, x):
        rows = paddle.assign(3)
        cols = paddle.assign(10)
        out = paddle.eye(rows, cols)
        return out

    def test_error(self):
        with self.assertRaises(TypeError):
            paddle.eye(-1)


class TestEyeFP16OP(TestEyeOp):
    '''Test eye op with specified dtype'''

    def init_dtype(self):
        self.dtype = np.float16


class TestEyeComplex64OP(TestEyeOp):
    '''Test eye op with specified dtype'''

    def init_dtype(self):
        self.dtype = np.complex64


class TestEyeComplex128OP(TestEyeOp):
    '''Test eye op with specified dtype'''

    def init_dtype(self):
        self.dtype = np.complex128


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestEyeBF16OP(OpTest):
    def setUp(self):
        self.op_type = "eye"
        self.dtype = np.uint16
        self.python_api = paddle.eye
        self.prim_op_type = "comp"
        self.public_python_api = paddle.eye
        self.inputs = {}
        self.attrs = {
            'num_rows': 219,
            'num_columns': 319,
        }
        self.outputs = {'Out': np.eye(219, 319)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True, check_prim_pir=True)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
