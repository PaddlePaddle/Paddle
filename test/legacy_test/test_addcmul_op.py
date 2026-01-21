#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    is_custom_device,
)

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class TestAddcmulOp(OpTest):
    # test basic functionality
    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.init_dtype_type()
        self.init_shapes_and_data()

        # Compute expected output: input + value * tensor1 * tensor2
        self.outputs = {
            'out': self.inputs['input']
            + self.attrs['value']
            * self.inputs['tensor1']
            * self.inputs['tensor2']
        }

    def init_dtype_type(self):
        self.dtype = np.float64

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((10, 20)).astype(self.dtype),
            'tensor1': np.random.random((10, 20)).astype(self.dtype),
            'tensor2': np.random.random((10, 20)).astype(self.dtype),
        }
        self.attrs = {'value': 0.5}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        # Test gradient for all three inputs
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )

    def test_check_grad_input(self):
        # Test gradient for input only
        self.check_grad(
            ['input'],
            'out',
            no_grad_set=None,
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )

    def test_check_grad_tensor1(self):
        # Test gradient for tensor1 only
        self.check_grad(
            ['tensor1'],
            'out',
            no_grad_set=None,
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )

    def test_check_grad_tensor2(self):
        # Test gradient for tensor2 only
        self.check_grad(
            ['tensor2'],
            'out',
            no_grad_set=None,
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )


class TestAddcmulOp2(TestAddcmulOp):
    # test with different value parameter
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((10, 20)).astype(self.dtype),
            'tensor1': np.random.random((10, 20)).astype(self.dtype),
            'tensor2': np.random.random((10, 20)).astype(self.dtype),
        }
        self.attrs = {'value': 2.0}


class TestAddcmulOp3(TestAddcmulOp):
    # test with negative value
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((8, 16)).astype(self.dtype),
            'tensor1': np.random.random((8, 16)).astype(self.dtype),
            'tensor2': np.random.random((8, 16)).astype(self.dtype),
        }
        self.attrs = {'value': -1.5}


class TestAddcmulOp4(TestAddcmulOp):
    # test with broadcasting (1D case)
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((100,)).astype(self.dtype),
            'tensor1': np.random.random((100,)).astype(self.dtype),
            'tensor2': np.random.random((100,)).astype(self.dtype),
        }
        self.attrs = {'value': 1.0}


class TestAddcmulOp5(TestAddcmulOp):
    # test with 3D tensors
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((4, 5, 6)).astype(self.dtype),
            'tensor1': np.random.random((4, 5, 6)).astype(self.dtype),
            'tensor2': np.random.random((4, 5, 6)).astype(self.dtype),
        }
        self.attrs = {'value': 0.1}


class TestAddcmulOp6(TestAddcmulOp):
    # test with large tensors
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((100, 100)).astype(self.dtype),
            'tensor1': np.random.random((100, 100)).astype(self.dtype),
            'tensor2': np.random.random((100, 100)).astype(self.dtype),
        }
        self.attrs = {'value': 0.5}


class TestAddcmulFP16Op(TestAddcmulOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2, check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            max_relative_error=0.01,
            check_pir=True,
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['input'],
            'out',
            max_relative_error=0.01,
            no_grad_set=None,
            check_pir=True,
        )

    def test_check_grad_tensor1(self):
        self.check_grad(
            ['tensor1'],
            'out',
            max_relative_error=0.01,
            no_grad_set=None,
            check_pir=True,
        )

    def test_check_grad_tensor2(self):
        self.check_grad(
            ['tensor2'],
            'out',
            max_relative_error=0.01,
            no_grad_set=None,
            check_pir=True,
        )


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestAddcmulBF16Op(OpTest):
    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.uint16

        input_np = np.random.random((10, 20)).astype(np.float32)
        tensor1_np = np.random.random((10, 20)).astype(np.float32)
        tensor2_np = np.random.random((10, 20)).astype(np.float32)
        value = 0.5

        self.inputs = {
            'input': convert_float_to_uint16(input_np),
            'tensor1': convert_float_to_uint16(tensor1_np),
            'tensor2': convert_float_to_uint16(tensor2_np),
        }
        self.attrs = {'value': value}
        self.outputs = {
            'out': convert_float_to_uint16(
                input_np + value * tensor1_np * tensor2_np
            )
        }

    def test_check_output(self):
        place = get_device_place()
        self.check_output_with_place(place, check_pir=True)

    def test_check_grad_normal(self):
        place = get_device_place()
        self.check_grad_with_place(
            place, ['input', 'tensor1', 'tensor2'], 'out', check_pir=True
        )

    def test_check_grad_input(self):
        place = get_device_place()
        self.check_grad_with_place(
            place, ['input'], 'out', no_grad_set=None, check_pir=True
        )

    def test_check_grad_tensor1(self):
        place = get_device_place()
        self.check_grad_with_place(
            place, ['tensor1'], 'out', no_grad_set=None, check_pir=True
        )

    def test_check_grad_tensor2(self):
        place = get_device_place()
        self.check_grad_with_place(
            place, ['tensor2'], 'out', no_grad_set=None, check_pir=True
        )


class TestAddcmulOpError(unittest.TestCase):
    # test error cases
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # Test with wrong dtype (int32)
            input = paddle.static.data(
                name='input',
                shape=[4, 4],
                dtype="int32",
            )
            x3 = paddle.static.data(name='x3', shape=[4, 4], dtype="int32")
            x4 = paddle.static.data(name='x4', shape=[4, 4], dtype="int32")
            self.assertRaises(TypeError, paddle.addcmul, input, x3, x4)
        paddle.disable_static()

    def test_broadcast_errors(self):
        # Test incompatible shapes
        paddle.disable_static()
        input = paddle.randn([2, 3])
        tensor1 = paddle.randn([2, 4])  # Incompatible shape
        tensor2 = paddle.randn([2, 3])
        with self.assertRaises((ValueError, RuntimeError)):
            paddle.addcmul(input, tensor1, tensor2)


class TestAddcmulAPI(unittest.TestCase):
    # Test Python API and compatibility
    def setUp(self):
        np.random.seed(123)
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.np_input = np.random.uniform(1, 4, self.shape).astype(self.dtype)
        self.np_tensor1 = np.random.uniform(1, 4, self.shape).astype(self.dtype)
        self.np_tensor2 = np.random.uniform(1, 4, self.shape).astype(self.dtype)

    def test_dygraph_compatibility(self):
        """测试动态图参数兼容性"""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        t1 = paddle.to_tensor(self.np_tensor1)
        t2 = paddle.to_tensor(self.np_tensor2)

        # 位置参数 (args)
        out1 = paddle.addcmul(x, t1, t2, value=0.5)

        # Paddle关键字参数 (kwargs)
        out2 = paddle.addcmul(input=x, tensor1=t1, tensor2=t2, value=0.5)

        # 测试默认值 value=1.0
        out3 = paddle.addcmul(x, t1, t2)

        # Tensor方法
        out4 = x.addcmul(t1, t2, value=0.5)

        # Numpy参考输出
        ref_out = self.np_input + 0.5 * self.np_tensor1 * self.np_tensor2
        ref_out_default = (
            self.np_input + 1.0 * self.np_tensor1 * self.np_tensor2
        )

        # 验证所有输出
        np.testing.assert_allclose(ref_out, out1.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out, out2.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out_default, out3.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out, out4.numpy(), rtol=1e-5)
        paddle.enable_static()

    def test_static_compatibility(self):
        """测试静态图参数兼容性"""
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            t1 = paddle.static.data(
                name="tensor1", shape=self.shape, dtype=self.dtype
            )
            t2 = paddle.static.data(
                name="tensor2", shape=self.shape, dtype=self.dtype
            )

            # 位置参数
            out1 = paddle.addcmul(x, t1, t2, value=0.5)

            # Paddle关键字参数
            out2 = paddle.addcmul(input=x, tensor1=t1, tensor2=t2, value=0.5)

            exe = base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_input,
                    "tensor1": self.np_tensor1,
                    "tensor2": self.np_tensor2,
                },
                fetch_list=[out1, out2],
            )
            ref_out = self.np_input + 0.5 * self.np_tensor1 * self.np_tensor2
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)
        paddle.disable_static()

    def test_edge_cases(self):
        """测试边界情况"""
        paddle.disable_static()

        # 测试 value=0 的情况
        input = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])

        result1 = paddle.addcmul(input, input, input, value=0)
        expected1 = input.numpy()
        np.testing.assert_allclose(result1.numpy(), expected1, rtol=1e-5)

        # 测试小数 value
        result2 = paddle.addcmul(input, input, input, value=0.01)
        expected2 = input.numpy() + 0.01 * input.numpy() * input.numpy()
        np.testing.assert_allclose(result2.numpy(), expected2, rtol=1e-5)

        # 测试负 value
        result3 = paddle.addcmul(input, input, input, value=-1.0)
        expected3 = input.numpy() - input.numpy() * input.numpy()
        np.testing.assert_allclose(result3.numpy(), expected3, rtol=1e-5)

        paddle.enable_static()

    def test_static_api(self):
        """测试静态图基本功能"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data('input', shape=[5, 10], dtype='float32')
            tensor1 = paddle.static.data(
                'tensor1', shape=[5, 10], dtype='float32'
            )
            tensor2 = paddle.static.data(
                'tensor2', shape=[5, 10], dtype='float32'
            )
            out = paddle.addcmul(input, tensor1, tensor2, value=0.5)

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)

            input_np = np.random.random((5, 10)).astype('float32')
            tensor1_np = np.random.random((5, 10)).astype('float32')
            tensor2_np = np.random.random((5, 10)).astype('float32')

            res = exe.run(
                feed={
                    'input': input_np,
                    'tensor1': tensor1_np,
                    'tensor2': tensor2_np,
                },
                fetch_list=[out],
            )
            expected = input_np + 0.5 * tensor1_np * tensor2_np
            np.testing.assert_allclose(res[0], expected, rtol=1e-5)
        paddle.disable_static()

    def test_broadcasting(self):
        """测试广播功能"""
        paddle.disable_static(self.place)

        # Test various broadcasting scenarios
        input_np = np.ones((3, 4), dtype='float32')
        tensor1_np = np.random.random((1, 4)).astype('float32')
        tensor2_np = np.random.random((3, 1)).astype('float32')
        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)
        out = paddle.addcmul(input, tensor1, tensor2, value=2.0)
        expected = input_np + 2.0 * (tensor1_np * tensor2_np)

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        # 测试复杂广播：1D 广播到 2D
        input = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        t1 = paddle.to_tensor([1.0, 2.0])
        t2 = paddle.to_tensor([[1.0], [2.0]])
        result = paddle.addcmul(input, t1, t2, value=0.5)
        expected = input.numpy() + 0.5 * t1.numpy() * t2.numpy()
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_torch_style_kwargs(self):
        """测试 Torch 风格的关键字参数"""
        paddle.disable_static(self.place)

        input_np = np.random.random((3, 4)).astype('float32')
        tensor1_np = np.random.random((3, 4)).astype('float32')
        tensor2_np = np.random.random((3, 4)).astype('float32')

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        # 使用 input 关键字 (Torch 风格)
        out = paddle.addcmul(
            input=input, tensor1=tensor1, tensor2=tensor2, value=0.5
        )
        expected = input_np + 0.5 * tensor1_np * tensor2_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_large_value(self):
        """测试大的 value 值"""
        paddle.disable_static(self.place)

        input_np = np.random.random((3, 4)).astype('float32')
        tensor1_np = np.random.random((3, 4)).astype('float32')
        tensor2_np = np.random.random((3, 4)).astype('float32')

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        out = paddle.addcmul(input, tensor1, tensor2, value=100.0)
        expected = input_np + 100.0 * tensor1_np * tensor2_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_tensor_method(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 10)).astype('float32')
        tensor1_np = np.random.random((5, 10)).astype('float32')
        tensor2_np = np.random.random((5, 10)).astype('float32')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        # Test Tensor.addcmul method
        out = input.addcmul(tensor1, tensor2, value=value)
        expected = input_np + value * tensor1_np * tensor2_np

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()


class TestAddcmulOp_ZeroSize(OpTest):
    # Test with zero-size tensors
    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.init_dtype_type()

        self.inputs = {
            'input': np.random.random((0, 5)).astype(self.dtype),
            'tensor1': np.random.random((0, 5)).astype(self.dtype),
            'tensor2': np.random.random((0, 5)).astype(self.dtype),
        }
        self.attrs = {'value': 0.5}
        self.outputs = {
            'out': self.inputs['input']
            + self.attrs['value']
            * self.inputs['tensor1']
            * self.inputs['tensor2']
        }

    def init_dtype_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(['input', 'tensor1', 'tensor2'], 'out', check_pir=True)


class TestAddcmulOp_4D(TestAddcmulOp):
    # test with 4D tensors
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((2, 3, 4, 5)).astype(self.dtype),
            'tensor1': np.random.random((2, 3, 4, 5)).astype(self.dtype),
            'tensor2': np.random.random((2, 3, 4, 5)).astype(self.dtype),
        }
        self.attrs = {'value': 0.5}


class TestAddcmulOp_5D(TestAddcmulOp):
    # test with 5D tensors
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor1': np.random.random((2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor2': np.random.random((2, 2, 3, 4, 5)).astype(self.dtype),
        }
        self.attrs = {'value': 0.3}


class TestAddcmulOp_6D(TestAddcmulOp):
    # test with 6D tensors
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((2, 2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor1': np.random.random((2, 2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor2': np.random.random((2, 2, 2, 3, 4, 5)).astype(self.dtype),
        }
        self.attrs = {'value': 0.2}


class TestAddcmulOp_0D(TestAddcmulOp):
    # test with 0D (scalar) tensors
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random(()).astype(self.dtype),
            'tensor1': np.random.random(()).astype(self.dtype),
            'tensor2': np.random.random(()).astype(self.dtype),
        }
        self.attrs = {'value': 0.5}


class TestAddcmulOp_ValueZero(TestAddcmulOp):
    # test with value=0
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((10, 10)).astype(self.dtype),
            'tensor1': np.random.random((10, 10)).astype(self.dtype),
            'tensor2': np.random.random((10, 10)).astype(self.dtype),
        }
        self.attrs = {'value': 0.0}


class TestAddcmulOp_ValueOne(TestAddcmulOp):
    # test with value=1.0 (default)
    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((10, 10)).astype(self.dtype),
            'tensor1': np.random.random((10, 10)).astype(self.dtype),
            'tensor2': np.random.random((10, 10)).astype(self.dtype),
        }
        self.attrs = {'value': 1.0}


class TestAddcmulOp_Float32(TestAddcmulOp):
    def init_dtype_type(self):
        self.dtype = np.float32

    def test_check_grad_normal(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.02,
            check_pir=True,
        )

    def test_check_grad_input(self):
        self.check_grad(
            ['input'],
            'out',
            no_grad_set=None,
            numeric_grad_delta=0.005,
            max_relative_error=0.02,
            check_pir=True,
        )

    def test_check_grad_tensor1(self):
        self.check_grad(
            ['tensor1'],
            'out',
            no_grad_set=None,
            numeric_grad_delta=0.005,
            max_relative_error=0.02,
            check_pir=True,
        )

    def test_check_grad_tensor2(self):
        self.check_grad(
            ['tensor2'],
            'out',
            no_grad_set=None,
            numeric_grad_delta=0.005,
            max_relative_error=0.02,
            check_pir=True,
        )


class TestAddcmulAPI_Out(unittest.TestCase):
    # Test out parameter
    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_out_parameter(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 10)).astype('float32')
        tensor1_np = np.random.random((5, 10)).astype('float32')
        tensor2_np = np.random.random((5, 10)).astype('float32')
        value = 0.5

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)
        out = paddle.empty([5, 10], dtype='float32')

        result = paddle.addcmul(input, tensor1, tensor2, value=value, out=out)
        expected = input_np + value * tensor1_np * tensor2_np

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_out_parameter_tensor_method(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 10)).astype('float32')
        tensor1_np = np.random.random((5, 10)).astype('float32')
        tensor2_np = np.random.random((5, 10)).astype('float32')
        value = 0.5

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        result = input.addcmul(tensor1, tensor2, value=value)
        expected = input_np + value * tensor1_np * tensor2_np

        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
        paddle.enable_static()


class TestAddcmulAPI_Gradient(unittest.TestCase):
    # Test gradient computation
    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_gradient(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((3, 4)).astype('float64')
        tensor1_np = np.random.random((3, 4)).astype('float64')
        tensor2_np = np.random.random((3, 4)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist
        self.assertIsNotNone(input.grad)
        self.assertIsNotNone(tensor1.grad)
        self.assertIsNotNone(tensor2.grad)

        # Verify gradient shapes
        self.assertEqual(input.grad.shape, list(input_np.shape))
        self.assertEqual(tensor1.grad.shape, list(tensor1_np.shape))
        self.assertEqual(tensor2.grad.shape, list(tensor2_np.shape))

        # Verify gradient values
        # d(out)/d(input) = 1
        np.testing.assert_allclose(
            input.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )
        # d(out)/d(tensor1) = value * tensor2
        np.testing.assert_allclose(
            tensor1.grad.numpy(), value * tensor2_np, rtol=1e-5
        )
        # d(out)/d(tensor2) = value * tensor1
        np.testing.assert_allclose(
            tensor2.grad.numpy(), value * tensor1_np, rtol=1e-5
        )

        paddle.enable_static()


class TestAddcmulAPI_DifferentDtypes(unittest.TestCase):
    # Test different data types
    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_float64(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 6)).astype('float64')
        tensor1_np = np.random.random((5, 6)).astype('float64')
        tensor2_np = np.random.random((5, 6)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        expected = input_np + value * tensor1_np * tensor2_np

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-10)
        paddle.enable_static()

    def test_float16(self):
        if not paddle.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.CUDAPlace(0))

        input_np = np.random.random((5, 6)).astype('float16')
        tensor1_np = np.random.random((5, 6)).astype('float16')
        tensor2_np = np.random.random((5, 6)).astype('float16')
        value = 0.5

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        expected = input_np.astype('float32') + value * tensor1_np.astype(
            'float32'
        ) * tensor2_np.astype('float32')

        np.testing.assert_allclose(
            out.numpy().astype('float32'), expected, rtol=1e-2, atol=1e-2
        )
        paddle.enable_static()


class TestAddcmulAPI_ErrorCases(unittest.TestCase):
    # Test error cases
    def test_dimension_mismatch(self):
        paddle.disable_static()
        input = paddle.randn([2, 3])
        tensor1 = paddle.randn([3, 4])  # Shape mismatch
        tensor2 = paddle.randn([2, 3])
        with self.assertRaises((ValueError, RuntimeError)):
            paddle.addcmul(input, tensor1, tensor2)
        paddle.enable_static()

    def test_rank_exceeds_6(self):
        paddle.disable_static()
        # Rank 7 should raise error
        input = paddle.randn([1, 1, 1, 1, 1, 1, 1])
        tensor1 = paddle.randn([1, 1, 1, 1, 1, 1, 1])
        tensor2 = paddle.randn([1, 1, 1, 1, 1, 1, 1])
        with self.assertRaises((ValueError, RuntimeError)):
            paddle.addcmul(input, tensor1, tensor2)
        paddle.enable_static()


class TestAddcmulAPI_Docstring(unittest.TestCase):
    # Test docstring existence
    def test_docstring(self):
        self.assertIsNotNone(paddle.addcmul.__doc__)


if __name__ == '__main__':
    unittest.main()
