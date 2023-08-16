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

import unittest

import numpy as np
from eager_op_test import OpTest, convert_float_to_uint16, paddle_static_guard

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class TestLinspaceOpCommonCase(OpTest):
    def setUp(self):
        self.op_type = "linspace"
        self.python_api = paddle.linspace
        self._set_dtype()
        self._set_data()
        self.attrs = {'dtype': self.attr_dtype}

    def _set_dtype(self):
        self.dtype = "float32"
        self.attr_dtype = int(core.VarDesc.VarType.FP32)

    def _set_data(self):
        self.outputs = {'Out': np.arange(0, 11).astype(self.dtype)}
        self.inputs = {
            'Start': np.array([0]).astype(self.dtype),
            'Stop': np.array([10]).astype(self.dtype),
            'Num': np.array([11]).astype('int32'),
        }

    def test_check_output(self):
        self.check_output()


class TestLinspaceOpReverseCase(TestLinspaceOpCommonCase):
    def _set_data(self):
        self.inputs = {
            'Start': np.array([10]).astype(self.dtype),
            'Stop': np.array([0]).astype(self.dtype),
            'Num': np.array([11]).astype('int32'),
        }
        self.outputs = {'Out': np.arange(10, -1, -1).astype(self.dtype)}

    def test_check_output(self):
        self.check_output()


class TestLinspaceOpNumOneCase(TestLinspaceOpCommonCase):
    def _set_data(self):
        self.inputs = {
            'Start': np.array([10]).astype(self.dtype),
            'Stop': np.array([0]).astype(self.dtype),
            'Num': np.array([1]).astype('int32'),
        }
        self.outputs = {'Out': np.array([10], dtype=self.dtype)}

    def test_check_output(self):
        self.check_output()


class TestLinspaceOpCommonCaseFP16(TestLinspaceOpCommonCase):
    def _set_dtype(self):
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)


class TestLinspaceOpReverseCaseFP16(TestLinspaceOpReverseCase):
    def _set_dtype(self):
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)


class TestLinspaceOpNumOneCaseFP16(TestLinspaceOpNumOneCase):
    def _set_dtype(self):
        self.dtype = np.float16
        self.attr_dtype = int(core.VarDesc.VarType.FP16)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    'not supported bf16',
)
class TestLinspaceOpCommonCaseBF16(TestLinspaceOpCommonCaseFP16):
    def _set_dtype(self):
        self.dtype = np.uint16
        self.attr_dtype = int(core.VarDesc.VarType.BF16)

    def _set_data(self):
        self.outputs = {
            'Out': convert_float_to_uint16(np.arange(0, 11).astype("float32"))
        }
        self.inputs = {
            'Start': convert_float_to_uint16(np.array([0]).astype("float32")),
            'Stop': convert_float_to_uint16(np.array([10]).astype("float32")),
            'Num': np.array([11]).astype('int32'),
        }

    def test_check_output(self):
        return self.check_output_with_place(core.CUDAPlace(0))


class TestLinspaceOpReverseCaseBF16(TestLinspaceOpCommonCaseBF16):
    def _set_data(self):
        self.inputs = {
            'Start': convert_float_to_uint16(np.array([10]).astype("float32")),
            'Stop': convert_float_to_uint16(np.array([0]).astype("float32")),
            'Num': np.array([11]).astype('int32'),
        }
        self.outputs = {
            'Out': convert_float_to_uint16(
                np.arange(10, -1, -1).astype("float32")
            )
        }


class TestLinspaceOpNumOneCaseBF16(TestLinspaceOpCommonCaseBF16):
    def _set_data(self):
        self.inputs = {
            'Start': convert_float_to_uint16(np.array([10]).astype("float32")),
            'Stop': convert_float_to_uint16(np.array([0]).astype("float32")),
            'Num': np.array([1]).astype('int32'),
        }
        self.outputs = {
            'Out': convert_float_to_uint16(np.array([10], dtype="float32"))
        }


class TestLinspaceAPI(unittest.TestCase):
    def test_variable_input1(self):
        with paddle_static_guard():
            start = paddle.full(shape=[1], fill_value=0, dtype='float32')
            stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
            num = paddle.full(shape=[1], fill_value=5, dtype='int32')
            out = paddle.linspace(start, stop, num, dtype='float32')
            exe = base.Executor(place=base.CPUPlace())
            res = exe.run(base.default_main_program(), fetch_list=[out])
            np_res = np.linspace(0, 10, 5, dtype='float32')
            self.assertEqual((res == np_res).all(), True)

    def test_variable_input2(self):
        start = paddle.full(shape=[1], fill_value=0, dtype='float32')
        stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
        num = paddle.full(shape=[1], fill_value=5, dtype='int32')
        out = paddle.linspace(start, stop, num, dtype='float32')
        np_res = np.linspace(0, 10, 5, dtype='float32')
        self.assertEqual((out.numpy() == np_res).all(), True)

    def test_dtype(self):
        with paddle_static_guard():
            out_1 = paddle.linspace(0, 10, 5, dtype='float32')
            out_2 = paddle.linspace(0, 10, 5, dtype=np.float32)
            out_3 = paddle.linspace(0, 10, 5, dtype=core.VarDesc.VarType.FP32)
            exe = base.Executor(place=base.CPUPlace())
            res_1, res_2, res_3 = exe.run(
                base.default_main_program(), fetch_list=[out_1, out_2, out_3]
            )
            np.testing.assert_array_equal(res_1, res_2)

    def test_name(self):
        with paddle_static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                out = paddle.linspace(
                    0, 10, 5, dtype='float32', name='linspace_res'
                )
                assert 'linspace_res' in out.name

    def test_imperative(self):
        out1 = paddle.linspace(0, 10, 5, dtype='float32')
        np_out1 = np.linspace(0, 10, 5, dtype='float32')
        out2 = paddle.linspace(0, 10, 5, dtype='int32')
        np_out2 = np.linspace(0, 10, 5, dtype='int32')
        out3 = paddle.linspace(0, 10, 200, dtype='int32')
        np_out3 = np.linspace(0, 10, 200, dtype='int32')
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)
        self.assertEqual((out3.numpy() == np_out3).all(), True)


class TestLinspaceOpError(unittest.TestCase):
    def test_errors(self):
        with paddle_static_guard():
            with program_guard(Program(), Program()):

                def test_dtype():
                    paddle.linspace(0, 10, 1, dtype="int8")

                self.assertRaises(TypeError, test_dtype)

                def test_dtype1():
                    paddle.linspace(0, 10, 1.33, dtype="int32")

                self.assertRaises(TypeError, test_dtype1)

                def test_start_type():
                    paddle.linspace([0], 10, 1, dtype="float32")

                self.assertRaises(TypeError, test_start_type)

                def test_end_type():
                    paddle.linspace(0, [10], 1, dtype="float32")

                self.assertRaises(TypeError, test_end_type)

                def test_step_dtype():
                    paddle.linspace(0, 10, [0], dtype="float32")

                self.assertRaises(TypeError, test_step_dtype)

                def test_start_dtype():
                    start = paddle.static.data(
                        shape=[1], dtype="float64", name="start"
                    )
                    paddle.linspace(start, 10, 1, dtype="float32")

                self.assertRaises(ValueError, test_start_dtype)

                def test_end_dtype():
                    end = paddle.static.data(
                        shape=[1], dtype="float64", name="end"
                    )
                    paddle.linspace(0, end, 1, dtype="float32")

                self.assertRaises(ValueError, test_end_dtype)

                def test_num_dtype():
                    num = paddle.static.data(
                        shape=[1], dtype="int32", name="step"
                    )
                    paddle.linspace(0, 10, num, dtype="float32")

                self.assertRaises(TypeError, test_step_dtype)


if __name__ == "__main__":
    unittest.main()
