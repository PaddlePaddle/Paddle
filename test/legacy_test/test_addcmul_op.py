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
    """Base test class for addcmul operator - 2D tensors"""

    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.init_dtype_type()
        self.init_shapes_and_data()

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

    def test_check_grad(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )


class TestAddcmulOp_1D(TestAddcmulOp):
    """Test 1D tensors - covers rank=1 branch"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((200,)).astype(self.dtype),
            'tensor1': np.random.random((200,)).astype(self.dtype),
            'tensor2': np.random.random((200,)).astype(self.dtype),
        }
        self.attrs = {'value': 1.5}


class TestAddcmulOp_3D(TestAddcmulOp):
    """Test 3D tensors - covers rank=3 branch"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((4, 5, 6)).astype(self.dtype),
            'tensor1': np.random.random((4, 5, 6)).astype(self.dtype),
            'tensor2': np.random.random((4, 5, 6)).astype(self.dtype),
        }
        self.attrs = {'value': 0.1}


class TestAddcmulOp_4D(TestAddcmulOp):
    """Test 4D tensors - covers rank=4 branch"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((2, 3, 4, 5)).astype(self.dtype),
            'tensor1': np.random.random((2, 3, 4, 5)).astype(self.dtype),
            'tensor2': np.random.random((2, 3, 4, 5)).astype(self.dtype),
        }
        self.attrs = {'value': 0.5}


class TestAddcmulOp_5D(TestAddcmulOp):
    """Test 5D tensors - covers rank=5 branch"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor1': np.random.random((2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor2': np.random.random((2, 2, 3, 4, 5)).astype(self.dtype),
        }
        self.attrs = {'value': 0.3}


class TestAddcmulOp_6D(TestAddcmulOp):
    """Test 6D tensors - covers rank=6 branch"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((2, 2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor1': np.random.random((2, 2, 2, 3, 4, 5)).astype(self.dtype),
            'tensor2': np.random.random((2, 2, 2, 3, 4, 5)).astype(self.dtype),
        }
        self.attrs = {'value': 0.2}


class TestAddcmulOp_0D(TestAddcmulOp):
    """Test 0D (scalar) tensors - covers rank=0 branch"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random(()).astype(self.dtype),
            'tensor1': np.random.random(()).astype(self.dtype),
            'tensor2': np.random.random(()).astype(self.dtype),
        }
        self.attrs = {'value': 0.5}


class TestAddcmulOp_NegativeValue(TestAddcmulOp):
    """Test with negative value"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((10, 10)).astype(self.dtype),
            'tensor1': np.random.random((10, 10)).astype(self.dtype),
            'tensor2': np.random.random((10, 10)).astype(self.dtype),
        }
        self.attrs = {'value': -1.5}


class TestAddcmulOp_ValueZero(TestAddcmulOp):
    """Test with value=0"""

    def init_shapes_and_data(self):
        self.inputs = {
            'input': np.random.random((10, 10)).astype(self.dtype),
            'tensor1': np.random.random((10, 10)).astype(self.dtype),
            'tensor2': np.random.random((10, 10)).astype(self.dtype),
        }
        self.attrs = {'value': 0.0}


class TestAddcmulFP16Op(TestAddcmulOp):
    """Test float16 dtype - output only, no grad check (requires fp64)"""

    no_need_check_grad = True

    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2, check_pir=True)

    def test_check_grad(self):
        # Skip grad check for fp16 - OpTest requires fp64 for gradient checking
        pass


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestAddcmulBF16Op(OpTest):
    """Test bfloat16 dtype"""

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

    def test_check_grad(self):
        place = get_device_place()
        self.check_grad_with_place(
            place, ['input', 'tensor1', 'tensor2'], 'out', check_pir=True
        )


class TestAddcmulOpError(unittest.TestCase):
    """Test error cases"""

    def test_type_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            input = paddle.static.data(
                name='input', shape=[4, 4], dtype="int32"
            )
            x3 = paddle.static.data(name='x3', shape=[4, 4], dtype="int32")
            x4 = paddle.static.data(name='x4', shape=[4, 4], dtype="int32")
            self.assertRaises(TypeError, paddle.addcmul, input, x3, x4)
        paddle.disable_static()

    def test_shape_errors(self):
        paddle.disable_static()
        input = paddle.randn([2, 3])
        tensor1 = paddle.randn([2, 4])
        tensor2 = paddle.randn([2, 3])
        with self.assertRaises((ValueError, RuntimeError)):
            paddle.addcmul(input, tensor1, tensor2)
        paddle.enable_static()

    def test_rank_exceeds_6(self):
        paddle.disable_static()
        input = paddle.randn([1, 1, 1, 1, 1, 1, 1])
        tensor1 = paddle.randn([1, 1, 1, 1, 1, 1, 1])
        tensor2 = paddle.randn([1, 1, 1, 1, 1, 1, 1])
        with self.assertRaises((ValueError, RuntimeError)):
            paddle.addcmul(input, tensor1, tensor2)
        paddle.enable_static()


class TestAddcmulAPI(unittest.TestCase):
    """Test Python API compatibility"""

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

    def test_dygraph_api(self):
        """Test dynamic graph API with args/kwargs"""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        t1 = paddle.to_tensor(self.np_tensor1)
        t2 = paddle.to_tensor(self.np_tensor2)

        # Positional args
        out1 = paddle.addcmul(x, t1, t2, value=0.5)
        # Keyword args
        out2 = paddle.addcmul(input=x, tensor1=t1, tensor2=t2, value=0.5)
        # Default value
        out3 = paddle.addcmul(x, t1, t2)
        # Tensor method
        out4 = x.addcmul(t1, t2, value=0.5)

        ref_out = self.np_input + 0.5 * self.np_tensor1 * self.np_tensor2
        ref_out_default = (
            self.np_input + 1.0 * self.np_tensor1 * self.np_tensor2
        )

        np.testing.assert_allclose(ref_out, out1.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out, out2.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out_default, out3.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out, out4.numpy(), rtol=1e-5)
        paddle.enable_static()

    def test_static_api(self):
        """Test static graph API"""
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
            out = paddle.addcmul(x, t1, t2, value=0.5)

            exe = base.Executor(paddle.CPUPlace())
            result = exe.run(
                main,
                feed={
                    "x": self.np_input,
                    "tensor1": self.np_tensor1,
                    "tensor2": self.np_tensor2,
                },
                fetch_list=[out],
            )
            ref_out = self.np_input + 0.5 * self.np_tensor1 * self.np_tensor2
            np.testing.assert_allclose(result[0], ref_out, rtol=1e-5)
        paddle.disable_static()

    def test_broadcasting(self):
        """Test broadcasting functionality"""
        paddle.disable_static(self.place)

        input_np = np.ones((3, 4), dtype='float32')
        tensor1_np = np.random.random((1, 4)).astype('float32')
        tensor2_np = np.random.random((3, 1)).astype('float32')
        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)
        out = paddle.addcmul(input, tensor1, tensor2, value=2.0)
        expected = input_np + 2.0 * (tensor1_np * tensor2_np)

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_edge_cases(self):
        """Test edge cases: value=0, negative value"""
        paddle.disable_static()
        input = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])

        # value=0
        result1 = paddle.addcmul(input, input, input, value=0)
        np.testing.assert_allclose(result1.numpy(), input.numpy(), rtol=1e-5)

        # negative value
        result2 = paddle.addcmul(input, input, input, value=-1.0)
        expected = input.numpy() - input.numpy() * input.numpy()
        np.testing.assert_allclose(result2.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_out_parameter(self):
        """Test out parameter"""
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 10)).astype('float32')
        tensor1_np = np.random.random((5, 10)).astype('float32')
        tensor2_np = np.random.random((5, 10)).astype('float32')

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)
        out = paddle.empty([5, 10], dtype='float32')

        result = paddle.addcmul(input, tensor1, tensor2, value=0.5, out=out)
        expected = input_np + 0.5 * tensor1_np * tensor2_np

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
        paddle.enable_static()


class TestAddcmulGradient(unittest.TestCase):
    """Test gradient computation through autograd"""

    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_gradient_2d(self):
        """Test gradient for 2D tensors"""
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

        # Verify gradient values
        np.testing.assert_allclose(
            input.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor1.grad.numpy(), value * tensor2_np, rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor2.grad.numpy(), value * tensor1_np, rtol=1e-5
        )

        paddle.enable_static()

    def test_gradient_0d(self):
        """Test gradient for 0D (scalar) tensors - covers GradFunctionZero"""
        paddle.disable_static(self.place)

        input_np = np.array(1.5).astype('float64')
        tensor1_np = np.array(2.0).astype('float64')
        tensor2_np = np.array(3.0).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        out.backward()

        np.testing.assert_allclose(input.grad.numpy(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(
            tensor1.grad.numpy(), value * tensor2_np, rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor2.grad.numpy(), value * tensor1_np, rtol=1e-5
        )

        paddle.enable_static()

    def test_gradient_broadcast(self):
        """Test gradient with broadcasting"""
        paddle.disable_static(self.place)

        input_np = np.random.random((3, 4)).astype('float64')
        tensor1_np = np.random.random((1, 4)).astype('float64')
        tensor2_np = np.random.random((3, 1)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        # Verify gradients exist and have correct shapes
        self.assertEqual(list(input.grad.shape), list(input_np.shape))
        self.assertEqual(list(tensor1.grad.shape), list(tensor1_np.shape))
        self.assertEqual(list(tensor2.grad.shape), list(tensor2_np.shape))

        paddle.enable_static()

    def test_gradient_3d(self):
        """Test gradient for 3D tensors"""
        paddle.disable_static(self.place)

        input_np = np.random.random((2, 3, 4)).astype('float64')
        tensor1_np = np.random.random((2, 3, 4)).astype('float64')
        tensor2_np = np.random.random((2, 3, 4)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        np.testing.assert_allclose(
            input.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor1.grad.numpy(), value * tensor2_np, rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor2.grad.numpy(), value * tensor1_np, rtol=1e-5
        )

        paddle.enable_static()

    def test_gradient_4d(self):
        """Test gradient for 4D tensors"""
        paddle.disable_static(self.place)

        input_np = np.random.random((2, 2, 3, 4)).astype('float64')
        tensor1_np = np.random.random((2, 2, 3, 4)).astype('float64')
        tensor2_np = np.random.random((2, 2, 3, 4)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        np.testing.assert_allclose(
            input.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )

        paddle.enable_static()

    def test_gradient_5d(self):
        """Test gradient for 5D tensors"""
        paddle.disable_static(self.place)

        input_np = np.random.random((2, 2, 2, 3, 4)).astype('float64')
        tensor1_np = np.random.random((2, 2, 2, 3, 4)).astype('float64')
        tensor2_np = np.random.random((2, 2, 2, 3, 4)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        np.testing.assert_allclose(
            input.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )

        paddle.enable_static()

    def test_gradient_6d(self):
        """Test gradient for 6D tensors"""
        paddle.disable_static(self.place)

        input_np = np.random.random((2, 2, 2, 2, 3, 4)).astype('float64')
        tensor1_np = np.random.random((2, 2, 2, 2, 3, 4)).astype('float64')
        tensor2_np = np.random.random((2, 2, 2, 2, 3, 4)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        np.testing.assert_allclose(
            input.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )

        paddle.enable_static()


class TestAddcmulDocstring(unittest.TestCase):
    """Test docstring existence"""

    def test_docstring(self):
        self.assertIsNotNone(paddle.addcmul.__doc__)


class TestAddcmulBroadcast2D(OpTest):
    """Test broadcasting with 2D tensors - covers GetBroadcastDims and ExtendDims2Rank"""

    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

        # input: (10, 100), tensor1: (10, 100), tensor2: (1, 100)
        input_np = np.random.random((10, 100)).astype(self.dtype)
        tensor1_np = np.random.random((10, 100)).astype(self.dtype)
        tensor2_np = np.random.random((1, 100)).astype(self.dtype)
        value = 0.5

        self.inputs = {
            'input': input_np,
            'tensor1': tensor1_np,
            'tensor2': tensor2_np,
        }
        self.attrs = {'value': value}
        self.outputs = {'out': input_np + value * tensor1_np * tensor2_np}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )


class TestAddcmulBroadcast3D(OpTest):
    """Test broadcasting with 3D tensors - different ndims"""

    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

        input_np = np.random.random((4, 10, 10)).astype(self.dtype)
        tensor1_np = np.random.random((4, 10, 10)).astype(self.dtype)
        tensor2_np = np.random.random((10, 10)).astype(self.dtype)
        value = 0.5

        self.inputs = {
            'input': input_np,
            'tensor1': tensor1_np,
            'tensor2': tensor2_np,
        }
        self.attrs = {'value': value}
        self.outputs = {'out': input_np + value * tensor1_np * tensor2_np}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )


class TestAddcmulBroadcast4D(OpTest):
    """Test broadcasting with 4D tensors - covers ExtendDims2Rank with different ranks"""

    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

        input_np = np.random.random((2, 3, 4, 10)).astype(self.dtype)
        tensor1_np = np.random.random((2, 3, 4, 10)).astype(self.dtype)
        tensor2_np = np.random.random((3, 4, 10)).astype(self.dtype)
        value = 0.5

        self.inputs = {
            'input': input_np,
            'tensor1': tensor1_np,
            'tensor2': tensor2_np,
        }
        self.attrs = {'value': value}
        self.outputs = {'out': input_np + value * tensor1_np * tensor2_np}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
        )


class TestAddcmulGradient1D(unittest.TestCase):
    """Test gradient for 1D tensors explicitly"""

    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_gradient_1d(self):
        """Test gradient for 1D tensors - covers rank=1 branch in grad kernel"""
        paddle.disable_static(self.place)

        input_np = np.random.random((100,)).astype('float64')
        tensor1_np = np.random.random((100,)).astype('float64')
        tensor2_np = np.random.random((100,)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        np.testing.assert_allclose(
            input.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor1.grad.numpy(), value * tensor2_np, rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor2.grad.numpy(), value * tensor1_np, rtol=1e-5
        )

        paddle.enable_static()


class TestAddcmulBroadcastGrad(unittest.TestCase):
    """Test gradient with broadcasting - covers reduce in grad kernel"""

    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_gradient_broadcast_2d(self):
        """Test gradient with 2D broadcast"""
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 6)).astype('float64')
        tensor1_np = np.random.random((1, 6)).astype('float64')
        tensor2_np = np.random.random((5, 1)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        # Verify gradients have correct shapes after broadcast reduction
        self.assertEqual(list(input.grad.shape), [5, 6])
        self.assertEqual(list(tensor1.grad.shape), [1, 6])
        self.assertEqual(list(tensor2.grad.shape), [5, 1])

        paddle.enable_static()

    def test_gradient_broadcast_3d(self):
        """Test gradient with 3D broadcast - different ndims"""
        paddle.disable_static(self.place)

        input_np = np.random.random((3, 4, 5)).astype('float64')
        tensor1_np = np.random.random((4, 5)).astype('float64')
        tensor2_np = np.random.random((1, 5)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        # Verify gradients have correct shapes
        self.assertEqual(list(input.grad.shape), [3, 4, 5])
        self.assertEqual(list(tensor1.grad.shape), [4, 5])
        self.assertEqual(list(tensor2.grad.shape), [1, 5])

        paddle.enable_static()

    def test_gradient_broadcast_4d(self):
        """Test gradient with 4D broadcast - covers ExtendDims2Rank in grad"""
        paddle.disable_static(self.place)

        input_np = np.random.random((2, 3, 4, 5)).astype('float64')
        tensor1_np = np.random.random((4, 5)).astype('float64')
        tensor2_np = np.random.random((3, 1, 5)).astype('float64')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        # Verify gradients have correct shapes
        self.assertEqual(list(input.grad.shape), [2, 3, 4, 5])
        self.assertEqual(list(tensor1.grad.shape), [4, 5])
        self.assertEqual(list(tensor2.grad.shape), [3, 1, 5])

        paddle.enable_static()


class TestAddcmulSymbolicShape(unittest.TestCase):
    """Test symbolic shape inference - covers multiary_infer_sym.cc"""

    def test_symbolic_shape_same_dims(self):
        """Test symbolic shape with same dimensions"""
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 3], dtype="float32")
            t1 = paddle.static.data(name="t1", shape=[2, 3], dtype="float32")
            t2 = paddle.static.data(name="t2", shape=[2, 3], dtype="float32")
            out = paddle.addcmul(x, t1, t2, value=0.5)
            self.assertEqual(list(out.shape), [2, 3])
        paddle.disable_static()

    def test_symbolic_shape_broadcast_xy(self):
        """Test symbolic shape with x > y dims - covers diffxy > 0 branch"""
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 3, 4], dtype="float32")
            t1 = paddle.static.data(name="t1", shape=[3, 4], dtype="float32")
            t2 = paddle.static.data(name="t2", shape=[2, 3, 4], dtype="float32")
            out = paddle.addcmul(x, t1, t2, value=0.5)
            self.assertEqual(list(out.shape), [2, 3, 4])
        paddle.disable_static()

    def test_symbolic_shape_broadcast_yx(self):
        """Test symbolic shape with y > x dims - covers diffxy < 0 branch"""
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4], dtype="float32")
            t1 = paddle.static.data(name="t1", shape=[2, 3, 4], dtype="float32")
            t2 = paddle.static.data(name="t2", shape=[2, 3, 4], dtype="float32")
            out = paddle.addcmul(x, t1, t2, value=0.5)
            self.assertEqual(list(out.shape), [2, 3, 4])
        paddle.disable_static()

    def test_symbolic_shape_broadcast_z(self):
        """Test symbolic shape with z > out1 dims - covers diffxyz > 0 branch"""
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4], dtype="float32")
            t1 = paddle.static.data(name="t1", shape=[3, 4], dtype="float32")
            t2 = paddle.static.data(name="t2", shape=[2, 3, 4], dtype="float32")
            out = paddle.addcmul(x, t1, t2, value=0.5)
            self.assertEqual(list(out.shape), [2, 3, 4])
        paddle.disable_static()

    def test_symbolic_shape_broadcast_out1(self):
        """Test symbolic shape with out1 > z dims - covers diffxyz < 0 branch"""
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 3, 4], dtype="float32")
            t1 = paddle.static.data(name="t1", shape=[2, 3, 4], dtype="float32")
            t2 = paddle.static.data(name="t2", shape=[4], dtype="float32")
            out = paddle.addcmul(x, t1, t2, value=0.5)
            self.assertEqual(list(out.shape), [2, 3, 4])
        paddle.disable_static()


class TestAddcmulBroadcastGradManual(unittest.TestCase):
    """Test broadcast gradient computation manually"""

    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_broadcast_2d(self):
        """Test 2D broadcast gradient"""
        paddle.disable_static(self.place)
        input_np = np.random.random((10, 12)).astype('float64')
        tensor1_np = np.random.random((1, 12)).astype('float64')
        tensor2_np = np.random.random((10, 1)).astype('float64')
        value = 0.5

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=value)
        loss = out.sum()
        loss.backward()

        self.assertEqual(list(input_t.grad.shape), [10, 12])
        self.assertEqual(list(tensor1_t.grad.shape), [1, 12])
        self.assertEqual(list(tensor2_t.grad.shape), [10, 1])
        paddle.enable_static()

    def test_broadcast_3d(self):
        """Test 3D broadcast gradient with different ndims"""
        paddle.disable_static(self.place)
        input_np = np.random.random((4, 5, 8)).astype('float64')
        tensor1_np = np.random.random((5, 8)).astype('float64')
        tensor2_np = np.random.random((1, 8)).astype('float64')
        value = 0.5

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=value)
        loss = out.sum()
        loss.backward()

        self.assertEqual(list(input_t.grad.shape), [4, 5, 8])
        self.assertEqual(list(tensor1_t.grad.shape), [5, 8])
        self.assertEqual(list(tensor2_t.grad.shape), [1, 8])
        paddle.enable_static()

    def test_broadcast_4d(self):
        """Test 4D broadcast gradient"""
        paddle.disable_static(self.place)
        input_np = np.random.random((2, 3, 4, 6)).astype('float64')
        tensor1_np = np.random.random((4, 6)).astype('float64')
        tensor2_np = np.random.random((3, 1, 6)).astype('float64')
        value = 0.5

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=value)
        loss = out.sum()
        loss.backward()

        self.assertEqual(list(input_t.grad.shape), [2, 3, 4, 6])
        self.assertEqual(list(tensor1_t.grad.shape), [4, 6])
        self.assertEqual(list(tensor2_t.grad.shape), [3, 1, 6])
        paddle.enable_static()


class TestAddcmulEmptyTensor(unittest.TestCase):
    """Test empty tensor handling"""

    def test_empty_output(self):
        paddle.disable_static()
        input = paddle.to_tensor(np.random.random((0, 4)).astype('float64'))
        tensor1 = paddle.to_tensor(np.random.random((0, 4)).astype('float64'))
        tensor2 = paddle.to_tensor(np.random.random((0, 4)).astype('float64'))
        out = paddle.addcmul(input, tensor1, tensor2, value=0.5)
        self.assertEqual(list(out.shape), [0, 4])
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
