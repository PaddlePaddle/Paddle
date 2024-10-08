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

import os
import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")
from op import Operator
from op_test import OpTest, convert_float_to_uint16, paddle_static_guard

import paddle
from paddle import base
from paddle.base import core


def fill_wrapper(shape, value=0.0):
    out = paddle.full(shape=shape, fill_value=value)
    return out


# Situation 1: Attr(shape) is a list(without tensor)
# Base case
class TestFillConstantOp(OpTest):
    def setUp(self):
        '''Test fill_constant op with default value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.init_dtype()
        self.init_shape()
        self.init_value()

        self.inputs = {}
        self.attrs = {'shape': self.shape, 'value': self.value}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.shape = [123, 92]

    def init_value(self):
        self.value = 0.0


class TestFillConstantFP32Op(TestFillConstantOp):
    '''Test fill_constant op with specified value'''

    def init_dtype(self):
        self.dtype = np.float32

    def init_value(self):
        self.value = 3.8


class TestFillConstantFP16Op(TestFillConstantOp):
    '''Test fill_constant op with specified value'''

    def init_dtype(self):
        self.dtype = np.float16

    def init_value(self):
        self.value = 3.8


class TestFillConstantINT64Op(TestFillConstantOp):
    '''Test fill_constant op with specified int64 value'''

    def init_dtype(self):
        self.dtype = np.int64

    def init_value(self):
        self.value = 10000000000


class TestFillConstantINT32Op(TestFillConstantOp):
    '''Test fill_constant op with specified int value'''

    def init_dtype(self):
        self.dtype = np.int32

    def init_value(self):
        self.value = 3


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFillConstantBF16Op(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.dtype = np.uint16
        self.inputs = {}
        self.attrs = {
            'shape': [123, 92],
            'value': 3.8,
            'dtype': core.VarDesc.VarType.BF16,
        }
        self.outputs = {'Out': convert_float_to_uint16(np.full((123, 92), 3.8))}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True)


class TestFillConstantOpWithSelectedRows(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()
        # create Out Variable
        out = scope.var('Out').get_selected_rows()

        # create and run fill_constant_op operator
        fill_constant_op = Operator(
            "fill_constant", shape=[123, 92], value=3.8, Out='Out'
        )
        fill_constant_op.run(scope, place)

        # get result from Out
        result_array = np.array(out.get_tensor())
        full_array = np.full((123, 92), 3.8, 'float32')

        np.testing.assert_array_equal(result_array, full_array)

    def test_fill_constant_with_selected_rows(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self.check_with_place(place)


# Situation 2: Attr(shape) is a list(with tensor)
class TestFillConstantOp1_ShapeTensorList(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.attrs = {'shape': self.infer_shape, 'value': self.value}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, 92]
        self.value = 3.8

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


class TestFillConstantOp2_ShapeTensorList(OpTest):
    def setUp(self):
        '''Test fill_constant op with default value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.attrs = {'shape': self.infer_shape}
        self.outputs = {'Out': np.full(self.shape, 0.0)}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, -1]

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


class TestFillConstantOp3_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.value = 10000000000


class TestFillConstantOp4_ShapeTensorList(TestFillConstantOp1_ShapeTensorList):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.value = 3


# Situation 3: shape is a tensor
class TestFillConstantOp1_ShapeTensor(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.init_data()

        self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
        self.attrs = {'value': self.value}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3.8

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


# Situation 4: value is a tensor
class TestFillConstantOp1_ValueTensor(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.init_data()

        self.inputs = {
            "ShapeTensor": np.array(self.shape).astype("int32"),
            'ValueTensor': np.array([self.value]).astype("float32"),
        }
        self.attrs = {'value': self.value + 1.0}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3.8
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


# Situation 5: value is a tensor
class TestFillConstantOp2_ValueTensor(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.init_data()

        self.inputs = {
            "ShapeTensor": np.array(self.shape).astype("int32"),
            'ValueTensor': np.array([self.value]).astype("int32"),
        }
        self.attrs = {'value': self.value, 'dtype': 2}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3
        self.dtype = np.int32

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


# Test python API
class TestFillConstantAPI(unittest.TestCase):

    def test_api(self):
        paddle.enable_static()
        positive_2_int32 = paddle.tensor.fill_constant([1], "int32", 2)
        positive_2_int64 = paddle.tensor.fill_constant([1], "int64", 2)

        shape_tensor_int32 = paddle.static.data(
            name="shape_tensor_int32", shape=[2], dtype="int32"
        )
        shape_tensor_int64 = paddle.static.data(
            name="shape_tensor_int64", shape=[2], dtype="int64"
        )

        out_1 = paddle.tensor.fill_constant(
            shape=[1, 2], dtype="float32", value=1.1
        )

        out_2 = paddle.tensor.fill_constant(
            shape=[1, positive_2_int32], dtype="float32", value=1.1
        )

        out_3 = paddle.tensor.fill_constant(
            shape=[1, positive_2_int64], dtype="float32", value=1.1
        )

        out_4 = paddle.tensor.fill_constant(
            shape=shape_tensor_int32, dtype="float32", value=1.1
        )

        out_5 = paddle.tensor.fill_constant(
            shape=shape_tensor_int64, dtype="float32", value=1.1
        )

        out_6 = paddle.tensor.fill_constant(
            shape=shape_tensor_int64, dtype=np.float32, value=1.1
        )

        val1 = paddle.tensor.fill_constant(
            shape=[1], dtype=np.float32, value=1.1
        )
        val2 = paddle.tensor.fill_constant(
            shape=[1], dtype=np.float64, value=1.1
        )
        out_7 = paddle.tensor.fill_constant(
            shape=shape_tensor_int64, dtype=np.float32, value=val1
        )

        out_8 = paddle.tensor.fill_constant(
            shape=shape_tensor_int64, dtype=np.float32, value=val2
        )

        exe = base.Executor(place=base.CPUPlace())
        res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8 = exe.run(
            paddle.static.default_main_program(),
            feed={
                "shape_tensor_int32": np.array([1, 2]).astype("int32"),
                "shape_tensor_int64": np.array([1, 2]).astype("int64"),
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8],
        )

        np.testing.assert_array_equal(
            res_1, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_2, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_3, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_4, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_5, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_6, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_7, np.full([1, 2], 1.1, dtype="float32")
        )
        np.testing.assert_array_equal(
            res_8, np.full([1, 2], 1.1, dtype="float32")
        )


class TestFillConstantImperative(unittest.TestCase):
    def test_api(self):
        with base.dygraph.guard():
            data1 = np.array([1, 2]).astype('int32')
            data2 = np.array([1.1]).astype('float32')
            data3 = np.array([88]).astype('int32')
            shape = paddle.to_tensor(data1)
            val = paddle.to_tensor(data2)
            value = paddle.to_tensor(data3)
            res1 = paddle.tensor.fill_constant(
                shape=[1, 2], dtype='float32', value=1.1
            )
            res2 = paddle.tensor.fill_constant(
                shape=shape, dtype='float32', value=1.1
            )
            res3 = paddle.tensor.fill_constant(
                shape=shape, dtype='float32', value=val
            )
            res4 = paddle.tensor.fill_constant(
                shape=shape, dtype='int32', value=value
            )
            np.testing.assert_array_equal(
                res1.numpy(), np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                res2.numpy(), np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                res3.numpy(), np.full([1, 2], 1.1, dtype="float32")
            )
            np.testing.assert_array_equal(
                res4.numpy(), np.full([1, 2], 88, dtype="int32")
            )

    def test_nan(self):
        with base.dygraph.guard():
            res = paddle.tensor.fill_constant([1], 'float32', np.nan)
            self.assertTrue(np.isnan(res.numpy().item(0)))

    def test_inf(self):
        with base.dygraph.guard():
            res = paddle.tensor.fill_constant([1], 'float32', np.inf)
            self.assertTrue(np.isinf(res.numpy().item(0)))

    def test_ninf(self):
        with base.dygraph.guard():
            res = paddle.tensor.fill_constant([1], 'float32', -np.inf)
            self.assertTrue(np.isinf(res.numpy().item(0)))
            self.assertEqual(-np.inf, res.numpy().item(0))


class TestFillConstantOpError(unittest.TestCase):

    def test_errors1(self):
        with paddle_static_guard(), paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # for ci coverage
            x1 = paddle.static.data(name='x1', shape=[-1, 1], dtype="int16")
            self.assertRaises(
                TypeError,
                paddle.tensor.fill_constant,
                shape=[1],
                value=5,
                dtype='uint4',
            )

            self.assertRaises(
                TypeError,
                paddle.tensor.fill_constant,
                shape=[1.1],
                value=5,
                dtype='float32',
                out=x1,
            )

            x3 = np.random.randn(100, 100).astype('int32')
            self.assertRaises(
                TypeError,
                paddle.tensor.fill_constant,
                shape=[100, 100],
                value=5,
                dtype='float64',
                out=x3,
            )

    def test_errors2(self):
        with paddle_static_guard(), paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The argument dtype of fill_constant_op must be one of bool, float16,
            # float32, float64, uint8, int16, int32 or int64
            x2 = paddle.static.data(name='x2', shape=[-1, 1], dtype="int32")
            self.assertRaises(
                TypeError,
                paddle.tensor.fill_constant,
                shape=[1],
                value=5,
                dtype='float64',
                out=x2,
            )

            # The shape dtype of fill_constant_op must be int32 or int64.
            def test_shape_tensor_dtype():
                shape = paddle.static.data(
                    name="shape_tensor", shape=[2], dtype="float32"
                )
                paddle.tensor.fill_constant(
                    shape=shape, dtype="float32", value=1
                )

            self.assertRaises(TypeError, test_shape_tensor_dtype)

            def test_shape_tensor_list_dtype():
                shape = paddle.static.data(
                    name="shape_tensor_list", shape=[1], dtype="bool"
                )
                paddle.tensor.fill_constant(
                    shape=[shape, 2], dtype="float32", value=1
                )

            self.assertRaises(TypeError, test_shape_tensor_list_dtype)

    def test_pir_errors(self):
        def test_shape_type():
            # The shape dtype of fill_constant_op must be int32 or int64.
            # test_shape_tensor_dtype:
            shape = paddle.static.data(
                name="shape_tensor", shape=[2], dtype="int32"
            )
            out = paddle.tensor.fill_constant(
                shape=shape, dtype="float32", value=1
            )
            exe = base.Executor(place=base.CPUPlace())
            exe.run(
                feed={"shape_tensor": np.array([1, 2]).astype("float32")},
                fetch_list=[out],
            )

        # TODO(chenzhiyang): pir test_shape_dtype


class TestFillConstantOp_ValueTensorBf16(OpTest):
    def setUp(self):
        '''Test fill_constant op with specified value'''
        self.op_type = "fill_constant"
        self.python_api = fill_wrapper
        self.init_data()

        self.inputs = {
            "ShapeTensor": np.array(self.shape).astype("int32"),
            'ValueTensor': convert_float_to_uint16(
                np.array([self.value]).astype("float32")
            ),
        }
        self.attrs = {'value': self.value, 'dtype': core.VarDesc.VarType.BF16}
        self.outputs = {'Out': np.full(self.shape, self.value)}

    def init_data(self):
        self.shape = [123, 92]
        self.value = 3.0
        self.dtype = np.uint16
        self.mkldnn_data_type = "bfloat16"

    def test_check_output(self):
        # no dynamic graph test for mkldnn
        self.check_output_with_place(
            core.CPUPlace(), check_dygraph=False, check_pir=False
        )


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
