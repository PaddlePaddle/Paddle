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
from paddle.base import core
from paddle.base.framework import Program, program_guard


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
        self.check_output(check_pir=True, check_cinn=True)

    def test_check_grad(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
            check_cinn=True,
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
    """Test 0D (scalar) tensors - covers rank=0 branch and AddcmulFunctionZero"""

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


class TestAddcmulFP16Op(TestAddcmulOp):
    """Test float16 dtype"""

    no_need_check_grad = True

    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-2, check_pir=True, check_cinn=True)

    def test_check_grad(self):
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


class TestAddcmulBroadcast2D(OpTest):
    """Test broadcasting - covers GetBroadcastDims and ExtendDims2Rank"""

    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

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
        self.check_output(check_pir=True, check_cinn=True)

    def test_check_grad(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
            check_cinn=True,
        )


class TestAddcmulBroadcast3D(OpTest):
    """Test broadcasting with different ndims"""

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
        self.check_output(check_pir=True, check_cinn=True)

    def test_check_grad(self):
        self.check_grad(
            ['input', 'tensor1', 'tensor2'],
            'out',
            numeric_grad_delta=0.005,
            max_relative_error=0.005,
            check_pir=True,
            check_cinn=True,
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


class TestAddcmulAPI(unittest.TestCase):
    """Test Python API compatibility"""

    def setUp(self):
        np.random.seed(123)
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.np_input = np.random.uniform(1, 4, self.shape).astype(self.dtype)
        self.np_tensor1 = np.random.uniform(1, 4, self.shape).astype(self.dtype)
        self.np_tensor2 = np.random.uniform(1, 4, self.shape).astype(self.dtype)

    def test_dygraph_api(self):
        """Test dynamic graph API"""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        t1 = paddle.to_tensor(self.np_tensor1)
        t2 = paddle.to_tensor(self.np_tensor2)

        out1 = paddle.addcmul(x, t1, t2, value=0.5)
        out2 = paddle.addcmul(input=x, tensor1=t1, tensor2=t2, value=0.5)
        out3 = x.addcmul(t1, t2, value=0.5)

        ref_out = self.np_input + 0.5 * self.np_tensor1 * self.np_tensor2
        np.testing.assert_allclose(ref_out, out1.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out, out2.numpy(), rtol=1e-5)
        np.testing.assert_allclose(ref_out, out3.numpy(), rtol=1e-5)
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

    def test_out_parameter(self):
        """Test out parameter"""
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        tensor1 = paddle.to_tensor(self.np_tensor1)
        tensor2 = paddle.to_tensor(self.np_tensor2)
        out = paddle.empty(self.shape, dtype=self.dtype)

        paddle.addcmul(input, tensor1, tensor2, value=0.5, out=out)
        expected = self.np_input + 0.5 * self.np_tensor1 * self.np_tensor2
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()


class TestAddcmulGradEmptyTensor(unittest.TestCase):
    """Test gradient with empty tensors - covers numel==0 branch"""

    def test_empty_grad(self):
        paddle.disable_static()
        input_t = paddle.to_tensor(
            np.random.random((0, 4)).astype('float64'), stop_gradient=False
        )
        tensor1_t = paddle.to_tensor(
            np.random.random((0, 4)).astype('float64'), stop_gradient=False
        )
        tensor2_t = paddle.to_tensor(
            np.random.random((0, 4)).astype('float64'), stop_gradient=False
        )

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.sum().backward()

        self.assertEqual(list(input_t.grad.shape), [0, 4])
        self.assertEqual(list(tensor1_t.grad.shape), [0, 4])
        self.assertEqual(list(tensor2_t.grad.shape), [0, 4])
        paddle.enable_static()


class TestAddcmulSelectiveGrad(unittest.TestCase):
    """Test gradient with selective stop_gradient - covers null grad pointer branches"""

    def test_only_input_grad(self):
        paddle.disable_static()
        input_np = np.random.random((3, 4)).astype('float64')
        tensor1_np = np.random.random((3, 4)).astype('float64')
        tensor2_np = np.random.random((3, 4)).astype('float64')

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=True)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=True)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.sum().backward()

        np.testing.assert_allclose(
            input_t.grad.numpy(), np.ones_like(input_np), rtol=1e-5
        )
        paddle.enable_static()

    def test_only_tensor1_grad(self):
        paddle.disable_static()
        input_np = np.random.random((3, 4)).astype('float64')
        tensor1_np = np.random.random((3, 4)).astype('float64')
        tensor2_np = np.random.random((3, 4)).astype('float64')

        input_t = paddle.to_tensor(input_np, stop_gradient=True)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=True)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.sum().backward()

        np.testing.assert_allclose(
            tensor1_t.grad.numpy(), 0.5 * tensor2_np, rtol=1e-5
        )
        paddle.enable_static()

    def test_only_tensor2_grad(self):
        paddle.disable_static()
        input_np = np.random.random((3, 4)).astype('float64')
        tensor1_np = np.random.random((3, 4)).astype('float64')
        tensor2_np = np.random.random((3, 4)).astype('float64')

        input_t = paddle.to_tensor(input_np, stop_gradient=True)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=True)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.sum().backward()

        np.testing.assert_allclose(
            tensor2_t.grad.numpy(), 0.5 * tensor1_np, rtol=1e-5
        )
        paddle.enable_static()


class TestAddcmulGrad0DScalar(unittest.TestCase):
    """Test 0D scalar gradient - covers AddcmulGradZero"""

    def test_0d_all_grads(self):
        paddle.disable_static()
        input_np = np.array(2.0).astype('float64')
        tensor1_np = np.array(3.0).astype('float64')
        tensor2_np = np.array(4.0).astype('float64')
        value = 0.5

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=value)
        out.backward()

        np.testing.assert_allclose(input_t.grad.numpy(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(
            tensor1_t.grad.numpy(), value * tensor2_np, rtol=1e-5
        )
        np.testing.assert_allclose(
            tensor2_t.grad.numpy(), value * tensor1_np, rtol=1e-5
        )
        paddle.enable_static()

    def test_0d_selective_grads(self):
        """Test 0D with selective gradients"""
        paddle.disable_static()
        input_t = paddle.to_tensor(
            np.array(2.0).astype('float64'), stop_gradient=True
        )
        tensor1_t = paddle.to_tensor(
            np.array(3.0).astype('float64'), stop_gradient=False
        )
        tensor2_t = paddle.to_tensor(
            np.array(4.0).astype('float64'), stop_gradient=True
        )

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.backward()

        np.testing.assert_allclose(tensor1_t.grad.numpy(), 0.5 * 4.0, rtol=1e-5)
        paddle.enable_static()


class TestAddcmulGradBroadcastReduction(unittest.TestCase):
    """Test gradient broadcast reduction - covers ReduceGrad and ComputeBroadcastGradDims"""

    def test_input_broadcast_reduce(self):
        """Test input gradient reduction"""
        paddle.disable_static()
        input_np = np.random.random((1, 4)).astype('float64')
        tensor1_np = np.random.random((3, 4)).astype('float64')
        tensor2_np = np.random.random((3, 4)).astype('float64')

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.sum().backward()

        self.assertEqual(list(input_t.grad.shape), [1, 4])
        np.testing.assert_allclose(
            input_t.grad.numpy(), np.ones((1, 4)) * 3, rtol=1e-5
        )
        paddle.enable_static()

    def test_tensor1_broadcast_reduce(self):
        """Test tensor1 gradient reduction"""
        paddle.disable_static()
        input_np = np.random.random((3, 4)).astype('float64')
        tensor1_np = np.random.random((1, 4)).astype('float64')
        tensor2_np = np.random.random((3, 4)).astype('float64')
        value = 0.5

        input_t = paddle.to_tensor(input_np, stop_gradient=True)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=True)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=value)
        out.sum().backward()

        self.assertEqual(list(tensor1_t.grad.shape), [1, 4])
        expected = (value * tensor2_np).sum(axis=0, keepdims=True)
        np.testing.assert_allclose(tensor1_t.grad.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_tensor2_broadcast_reduce_3d(self):
        """Test tensor2 gradient reduction with 3D"""
        paddle.disable_static()
        input_np = np.random.random((2, 3, 5)).astype('float64')
        tensor1_np = np.random.random((2, 3, 5)).astype('float64')
        tensor2_np = np.random.random((1, 1, 5)).astype('float64')
        value = 0.5

        input_t = paddle.to_tensor(input_np, stop_gradient=True)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=True)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=value)
        out.sum().backward()

        self.assertEqual(list(tensor2_t.grad.shape), [1, 1, 5])
        expected = (value * tensor1_np).sum(axis=(0, 1), keepdims=True)
        np.testing.assert_allclose(tensor2_t.grad.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_all_broadcast_reduce_4d(self):
        """Test all gradients with reduction in 4D"""
        paddle.disable_static()
        input_np = np.random.random((1, 1, 4, 5)).astype('float64')
        tensor1_np = np.random.random((2, 1, 1, 5)).astype('float64')
        tensor2_np = np.random.random((1, 3, 1, 1)).astype('float64')

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        self.assertEqual(list(out.shape), [2, 3, 4, 5])
        out.sum().backward()

        self.assertEqual(list(input_t.grad.shape), [1, 1, 4, 5])
        self.assertEqual(list(tensor1_t.grad.shape), [2, 1, 1, 5])
        self.assertEqual(list(tensor2_t.grad.shape), [1, 3, 1, 1])
        paddle.enable_static()


class TestAddcmulBroadcastGradHighRank(unittest.TestCase):
    """Test broadcast gradient for high rank tensors (5D, 6D)"""

    def test_broadcast_5d(self):
        paddle.disable_static()
        input_np = np.random.random((2, 2, 3, 4, 6)).astype('float64')
        tensor1_np = np.random.random((3, 4, 6)).astype('float64')
        tensor2_np = np.random.random((1, 1, 1, 4, 6)).astype('float64')

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.sum().backward()

        self.assertEqual(list(input_t.grad.shape), [2, 2, 3, 4, 6])
        self.assertEqual(list(tensor1_t.grad.shape), [3, 4, 6])
        self.assertEqual(list(tensor2_t.grad.shape), [1, 1, 1, 4, 6])
        paddle.enable_static()

    def test_broadcast_6d(self):
        paddle.disable_static()
        input_np = np.random.random((2, 2, 2, 3, 4, 5)).astype('float64')
        tensor1_np = np.random.random((2, 3, 4, 5)).astype('float64')
        tensor2_np = np.random.random((1, 1, 2, 1, 4, 5)).astype('float64')

        input_t = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1_t = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2_t = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input_t, tensor1_t, tensor2_t, value=0.5)
        out.sum().backward()

        self.assertEqual(list(input_t.grad.shape), [2, 2, 2, 3, 4, 5])
        self.assertEqual(list(tensor1_t.grad.shape), [2, 3, 4, 5])
        self.assertEqual(list(tensor2_t.grad.shape), [1, 1, 2, 1, 4, 5])
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
