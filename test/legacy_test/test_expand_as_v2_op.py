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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core


class TestExpandAsBasic(OpTest):
    def setUp(self):
        self.op_type = "expand_as_v2"
        self.prim_op_type = "comp"
        self.python_api = paddle.expand_as
        self.public_python_api = paddle.expand_as
        self.init_dtype()
        self.init_inputs_and_outputs()
        self.if_enable_cinn()

    def init_dtype(self):
        self.dtype = np.float64

    def init_inputs_and_outputs(self):
        x = np.random.rand(100).astype(self.dtype)
        target_tensor = np.random.rand(2, 100).astype(self.dtype)
        self.inputs = {'X': x, "Y": target_tensor}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [2, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(check_prim=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True, check_pir=True)


class TestExpandAs_ZeroDim1(TestExpandAsBasic):
    def init_inputs_and_outputs(self):
        x = np.random.random(()).astype(self.dtype)
        target_tensor = np.random.random(1).astype(self.dtype)
        self.inputs = {'X': x, "Y": target_tensor}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}


class TestExpandAs_ZeroDim2(TestExpandAsBasic):
    def init_inputs_and_outputs(self):
        x = np.random.random(()).astype(self.dtype)
        target_tensor = np.random.random(()).astype(self.dtype)
        self.inputs = {'X': x, "Y": target_tensor}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = []
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def if_enable_cinn(self):
        self.enable_cinn = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestExpandAsBasicBFP16OP(TestExpandAsBasic):
    def init_dtype(self):
        self.dtype = np.uint16

    def init_inputs_and_outputs(self):
        x = np.random.rand(100).astype(np.float32)
        target_tensor = np.random.rand(2, 100).astype(np.float32)
        self.inputs = {
            'X': convert_float_to_uint16(x),
            "Y": convert_float_to_uint16(target_tensor),
        }
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [2, 1]
        output = np.tile(x, bcast_dims)
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0), ['X'], 'Out', check_prim=True, check_pir=True
        )


class TestExpandAsOpRank2(TestExpandAsBasic):
    def init_inputs_and_outputs(self):
        x = np.random.rand(10, 12).astype(self.dtype)
        target_tensor = np.random.rand(10, 12).astype(self.dtype)
        self.inputs = {'X': x, "Y": target_tensor}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestExpandAsOpRank2BFP16OP(TestExpandAsBasicBFP16OP):
    def init_inputs_and_outputs(self):
        x = np.random.rand(10, 12).astype(np.float32)
        target_tensor = np.random.rand(10, 12).astype(np.float32)
        self.inputs = {
            'X': convert_float_to_uint16(x),
            "Y": convert_float_to_uint16(target_tensor),
        }
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1]
        output = np.tile(x, bcast_dims)
        self.outputs = {'Out': convert_float_to_uint16(output)}


class TestExpandAsOpRank3(TestExpandAsBasic):
    def init_inputs_and_outputs(self):
        x = np.random.rand(2, 3, 20).astype(self.dtype)
        target_tensor = np.random.rand(2, 3, 20).astype(self.dtype)
        self.inputs = {'X': x, "Y": target_tensor}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestExpandAsOpRank3BFP16OP(TestExpandAsBasicBFP16OP):
    def init_inputs_and_outputs(self):
        x = np.random.rand(2, 3, 20).astype(np.float32)
        target_tensor = np.random.rand(2, 3, 20).astype(np.float32)
        self.inputs = {
            'X': convert_float_to_uint16(x),
            "Y": convert_float_to_uint16(target_tensor),
        }
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [1, 1, 1]
        output = np.tile(x, bcast_dims)
        self.outputs = {'Out': convert_float_to_uint16(output)}


class TestExpandAsOpRank4(TestExpandAsBasic):
    def init_inputs_and_outputs(self):
        x = np.random.rand(1, 1, 7, 16).astype(self.dtype)
        target_tensor = np.random.rand(4, 6, 7, 16).astype(self.dtype)
        self.inputs = {'X': x, "Y": target_tensor}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [4, 6, 1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestExpandAsOpRank4BFP16OP(TestExpandAsBasicBFP16OP):
    def init_inputs_and_outputs(self):
        x = np.random.rand(1, 1, 7, 16).astype(np.float32)
        target_tensor = np.random.rand(4, 6, 7, 16).astype(np.float32)
        self.inputs = {
            'X': convert_float_to_uint16(x),
            "Y": convert_float_to_uint16(target_tensor),
        }
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [4, 6, 1, 1]
        output = np.tile(x, bcast_dims)
        self.outputs = {'Out': convert_float_to_uint16(output)}


class TestExpandAsOpRank5(TestExpandAsBasic):
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "expand_as_v2"
        self.prim_op_type = "comp"
        self.python_api = paddle.expand_as
        self.public_python_api = paddle.expand_as
        x = np.random.rand(1, 1, 7, 16).astype("int64")
        target_tensor = np.random.rand(4, 6, 7, 16).astype("float64")
        self.inputs = {'X': x, "Y": target_tensor}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [4, 6, 1, 1]
        output = np.tile(self.inputs['X'], bcast_dims)
        self.outputs = {'Out': output}

    def test_check_grad(self):
        pass


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestExpandAsOpRank5BFP16OP(TestExpandAsOpRank5):
    def setUp(self):
        self.op_type = "expand_as_v2"
        self.prim_op_type = "comp"
        self.python_api = paddle.expand_as
        self.public_python_api = paddle.expand_as
        x = np.random.rand(1, 1, 7, 16).astype("int64")
        target_tensor = np.random.rand(4, 6, 7, 16).astype("float32")
        self.inputs = {'X': x, "Y": convert_float_to_uint16(target_tensor)}
        self.attrs = {'target_shape': target_tensor.shape}
        bcast_dims = [4, 6, 1, 1]
        output = np.tile(x, bcast_dims)
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        pass


class TestExpandAsV2Error(unittest.TestCase):
    def test_errors(self):
        with base.program_guard(base.Program(), base.Program()):
            x1 = paddle.static.data(name='x1', shape=[-1, 4], dtype="uint8")
            x2 = paddle.static.data(name='x2', shape=[-1, 4], dtype="int32")
            self.assertRaises(TypeError, paddle.tensor.expand_as, x1, x2)
            x3 = paddle.static.data(name='x3', shape=[-1, 4], dtype="bool")
            x3.stop_gradient = False
            self.assertRaises(ValueError, paddle.tensor.expand_as, x3, x2)


# Test python API
class TestExpandAsV2API(unittest.TestCase):

    def test_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            input1 = np.random.random([12, 14]).astype("float32")
            input2 = np.random.random([2, 12, 14]).astype("float32")
            x = paddle.static.data(name='x', shape=[12, 14], dtype="float32")

            y = paddle.static.data(
                name='target_tensor',
                shape=[2, 12, 14],
                dtype="float32",
            )

            out_1 = paddle.expand_as(x, y=y)

            exe = base.Executor(place=base.CPUPlace())
            res_1 = exe.run(
                paddle.static.default_main_program(),
                feed={"x": input1, "target_tensor": input2},
                fetch_list=[out_1],
            )
            np.testing.assert_array_equal(res_1[0], np.tile(input1, (2, 1, 1)))


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
