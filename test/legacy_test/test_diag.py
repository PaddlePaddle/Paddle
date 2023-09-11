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

import unittest

import numpy as np
from eager_op_test import OpTest

import paddle
from paddle.base import Program, core, program_guard


class TestDiagOp(OpTest):
    def setUp(self):
        self.op_type = "diag"
        self.init_config()
        self.inputs = {'Diagonal': self.case}

        self.outputs = {'Out': np.diag(self.inputs['Diagonal'])}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output()

    def init_config(self):
        self.case = np.arange(3, 6)


class TestDiagOpCase1(TestDiagOp):
    def init_config(self):
        self.case = np.array([3], dtype='int32')


class TestDiagOpFp16(unittest.TestCase):
    def test_fp16(self):
        x_np = np.array([3], dtype='float16')
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(shape=[1, 0], name='x', dtype='float16')
            out = paddle.diag(x)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                exe.run(paddle.static.default_startup_program())
                out = exe.run(feed={'x': x_np}, fetch_list=[out])


class TestDiagError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):

            def test_diag_type():
                return paddle.diag(x=[1, 2, 3])

            self.assertRaises(TypeError, test_diag_type)


if __name__ == "__main__":
    unittest.main()
