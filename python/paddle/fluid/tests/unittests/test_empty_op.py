#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
from op_test import OpTest
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_


# Situation 1: Attr(shape) is a list(without tensor)
class TestEmptyOp(OpTest):

    def setUp(self):
        self.op_type = "empty"
        self.init_config()

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        data_type = outs[0].dtype
        if data_type in ['float32', 'float64', 'int32', 'int64']:
            max_value = np.nanmax(outs[0])
            min_value = np.nanmin(outs[0])

            always_full_zero = max_value == 0.0 and min_value == 0.0
            always_non_full_zero = max_value >= min_value
            self.assertTrue(always_full_zero or always_non_full_zero,
                            'always_full_zero or always_non_full_zero.')
        elif data_type in ['bool']:
            total_num = outs[0].size
            true_num = np.sum(outs[0] == True)
            false_num = np.sum(outs[0] == False)
            self.assertTrue(total_num == true_num + false_num,
                            'The value should always be True or False.')
        else:
            self.assertTrue(False, 'invalid data type')

    def init_config(self):
        shape = [500, 3]
        dtype = 'float32'
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyOp2(TestEmptyOp):

    def init_config(self):
        shape = [500, 3]
        dtype = 'float64'
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyOp3(TestEmptyOp):

    def init_config(self):
        shape = [500, 3]
        dtype = 'int32'
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyOp4(TestEmptyOp):

    def init_config(self):
        shape = [500, 3]
        dtype = 'int64'
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


class TestEmptyOp5(TestEmptyOp):

    def init_config(self):
        shape = [500, 3]
        dtype = 'bool'
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        self.attrs = {'shape': shape, 'dtype': dtype_inner}
        self.inputs = {}
        self.outputs = {'Out': np.zeros(shape).astype(dtype)}


# Situation 2: shape is a tensor
class TestEmptyOp_ShapeTensor(OpTest):

    def setUp(self):
        self.op_type = "empty"
        self.init_config()

    def init_config(self):
        self.shape = [500, 3]
        dtype = 'float32'
        dtype_inner = convert_np_dtype_to_dtype_(dtype)
        self.attrs = {'dtype': dtype_inner}
        self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
        self.outputs = {'Out': np.zeros(self.shape).astype(dtype)}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        data_type = outs[0].dtype
        if data_type in ['float32', 'float64', 'int32', 'int64']:
            max_value = np.nanmax(outs[0])
            min_value = np.nanmin(outs[0])

            always_full_zero = max_value == 0.0 and min_value == 0.0
            always_non_full_zero = max_value >= min_value
            self.assertTrue(always_full_zero or always_non_full_zero,
                            'always_full_zero or always_non_full_zero.')
        elif data_type in ['bool']:
            total_num = outs[0].size
            true_num = np.sum(outs[0] == True)
            false_num = np.sum(outs[0] == False)
            self.assertTrue(total_num == true_num + false_num,
                            'The value should always be True or False.')
        else:
            self.assertTrue(False, 'invalid data type')


# Situation 3: Attr(shape) is a list(with tensor)
class TestEmptyOp_ShapeTensorList(OpTest):

    def setUp(self):
        self.op_type = "empty"
        self.init_config()

    def init_config(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, 92]

        dtype = 'float32'
        dtype_inner = convert_np_dtype_to_dtype_(dtype)

        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.attrs = {'shape': self.infer_shape, 'dtype': dtype_inner}
        self.outputs = {'Out': np.zeros(self.shape).astype(dtype)}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        data_type = outs[0].dtype
        if data_type in ['float32', 'float64', 'int32', 'int64']:
            max_value = np.nanmax(outs[0])
            min_value = np.nanmin(outs[0])

            always_full_zero = max_value == 0.0 and min_value == 0.0
            always_non_full_zero = max_value >= min_value
            self.assertTrue(always_full_zero or always_non_full_zero,
                            'always_full_zero or always_non_full_zero.')
        elif data_type in ['bool']:
            total_num = outs[0].size
            true_num = np.sum(outs[0] == True)
            false_num = np.sum(outs[0] == False)
            self.assertTrue(total_num == true_num + false_num,
                            'The value should always be True or False.')
        else:
            self.assertTrue(False, 'invalid data type')


class TestEmptyAPI(unittest.TestCase):

    def __check_out__(self, out, dtype='float32'):
        max_value = np.nanmax(np.array(out))
        min_value = np.nanmin(np.array(out))
        always_non_full_zero = max_value >= min_value
        always_full_zero = max_value == 0.0 and min_value == 0.0
        self.assertTrue(always_full_zero or always_non_full_zero,
                        'always_full_zero or always_non_full_zero.')

    def test_dygraph_api_out(self):
        paddle.disable_static()
        shape = [200, 3]
        out = paddle.empty(shape=shape)
        self.__check_out__(out)
        paddle.enable_static()

    def test_dygraph_api_out_2(self):
        paddle.disable_static()
        shape_data = np.array([200, 3]).astype('int32')
        shape = paddle.to_tensor(shape_data)
        out = paddle.empty(shape=shape)
        self.__check_out__(out)
        paddle.enable_static()

    def test_dygraph_api_out_3(self):
        paddle.disable_static()
        shape_data = np.array([200, 3]).astype('int64')
        shape = paddle.to_tensor(shape_data)
        out = paddle.empty(shape=shape)
        self.__check_out__(out)
        paddle.enable_static()

    def test_dygraph_api_attr(self):
        paddle.disable_static()
        shape = [200, 3]
        dtype = 'float64'
        out = paddle.empty(shape=shape, dtype=dtype)
        self.__check_out__(out, dtype)
        paddle.enable_static()

    def test_static_graph(self):
        dtype = 'float64'

        positive_2_int32 = fluid.layers.fill_constant([1], "int32", 3)
        positive_2_int64 = fluid.layers.fill_constant([1], "int64", 3)

        shape_tensor_int32 = fluid.data(name="shape_tensor_int32",
                                        shape=[2],
                                        dtype="int32")
        shape_tensor_int64 = fluid.data(name="shape_tensor_int64",
                                        shape=[2],
                                        dtype="int64")
        shape_tensor_unknown = fluid.data(name="shape_tensor_unknown",
                                          shape=[-1],
                                          dtype="int64")

        out_1 = paddle.empty(shape=[200, 3], dtype=dtype)
        out_2 = paddle.empty(shape=shape_tensor_int32, dtype=dtype)
        out_3 = paddle.empty(shape=shape_tensor_int64, dtype=dtype)
        out_4 = paddle.empty(shape=[200, positive_2_int32], dtype=dtype)
        out_5 = paddle.empty(shape=[200, positive_2_int64], dtype=dtype)
        out_6 = paddle.empty(shape=shape_tensor_unknown, dtype=dtype)

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        res_1, res_2, res_3, res_4, res_5, res_6 = exe.run(
            fluid.default_main_program(),
            feed={
                "shape_tensor_int32": np.array([200, 3]).astype("int32"),
                "shape_tensor_int64": np.array([200, 3]).astype("int64"),
                "shape_tensor_unknown": np.array([200, 3]).astype("int64"),
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6])

        self.__check_out__(res_1, dtype)
        self.__check_out__(res_2, dtype)
        self.__check_out__(res_3, dtype)
        self.__check_out__(res_4, dtype)
        self.__check_out__(res_5, dtype)
        self.__check_out__(res_6, dtype)


class TestEmptyError(unittest.TestCase):

    def test_attr(self):

        def test_dtype():
            shape = [200, 3]
            dtype = 'uint8'
            result = paddle.empty(shape=shape, dtype=dtype)

        self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    unittest.main()
