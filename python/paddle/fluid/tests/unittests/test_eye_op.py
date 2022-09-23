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

from __future__ import print_function

import os
import unittest
import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework

from paddle.fluid.framework import program_guard, Program
from test_attribute_var import UnittestBase


class TestEyeOp(OpTest):

    def setUp(self):
        '''
        Test eye op with specified shape
        '''
        self.python_api = paddle.eye
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {
            'num_rows': 219,
            'num_columns': 319,
            'dtype': framework.convert_np_dtype_to_dtype_(np.int32)
        }
        self.outputs = {'Out': np.eye(219, 319, dtype=np.int32)}

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestEyeOp1(OpTest):

    def setUp(self):
        '''
        Test eye op with default parameters
        '''
        self.python_api = paddle.eye
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {'num_rows': 50}
        self.outputs = {'Out': np.eye(50, dtype=float)}

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestEyeOp2(OpTest):

    def setUp(self):
        '''
        Test eye op with specified shape
        '''
        self.python_api = paddle.eye
        self.op_type = "eye"

        self.inputs = {}
        self.attrs = {'num_rows': 99, 'num_columns': 1}
        self.outputs = {'Out': np.eye(99, 1, dtype=float)}

    def test_check_output(self):
        self.check_output(check_eager=True)


class API_TestTensorEye(unittest.TestCase):

    def test_out(self):
        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10)
            place = fluid.CPUPlace()
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, num_columns=7, dtype="float64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, 7, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            data = paddle.eye(10, dtype="int64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result, = exe.run(fetch_list=[data])
            expected_result = np.eye(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)

        paddle.disable_static()
        out = paddle.eye(10, dtype="int64")
        expected_result = np.eye(10, dtype="int64")
        paddle.enable_static()
        self.assertEqual((out.numpy() == expected_result).all(), True)

        paddle.disable_static()
        batch_shape = [2]
        out = fluid.layers.eye(10, 10, dtype="int64", batch_shape=batch_shape)
        result = np.eye(10, dtype="int64")
        expected_result = []
        for index in reversed(batch_shape):
            tmp_result = []
            for i in range(index):
                tmp_result.append(result)
            result = tmp_result
            expected_result = np.stack(result, axis=0)
        paddle.enable_static()
        self.assertEqual(out.numpy().shape == np.array(expected_result).shape,
                         True)
        self.assertEqual((out.numpy() == expected_result).all(), True)

        paddle.disable_static()
        batch_shape = [3, 2]
        out = fluid.layers.eye(10, 10, dtype="int64", batch_shape=batch_shape)
        result = np.eye(10, dtype="int64")
        expected_result = []
        for index in reversed(batch_shape):
            tmp_result = []
            for i in range(index):
                tmp_result.append(result)
            result = tmp_result
            expected_result = np.stack(result, axis=0)
        paddle.enable_static()
        self.assertEqual(out.numpy().shape == np.array(expected_result).shape,
                         True)
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
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]

            tmp = self.call_func(feat)
            out = feat + tmp

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue(self.var_prefix() in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[tmp, out])
            gt = np.eye(3, 10)
            np.testing.assert_allclose(res[0], gt)
            paddle.static.save_inference_model(self.save_path, [x], [tmp, out],
                                               exe)
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


class TestEyeRowsCol2(TestEyeRowsCol):

    def call_func(self, x):
        rows = paddle.assign(3)
        cols = paddle.assign(10)
        out = paddle.fluid.layers.eye(rows, cols)
        return out

    def test_error(self):
        with self.assertRaises(TypeError):
            paddle.fluid.layers.eye(-1)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
