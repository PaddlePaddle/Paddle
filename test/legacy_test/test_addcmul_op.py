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
)

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import Program, program_guard


class TestAddcmulOp(OpTest):
    """Base test class for addcmul operator - 2D tensors"""

    def setUp(self):
        self.op_type = "addcmul"
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


class TestAddcmulOp_Int32(OpTest):
    """Test int32 dtype - aligned with PyTorch integer type support"""

    def setUp(self):
        self.op_type = "addcmul"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.int32

        input_np = np.random.randint(1, 10, (10, 20)).astype(self.dtype)
        tensor1_np = np.random.randint(1, 10, (10, 20)).astype(self.dtype)
        tensor2_np = np.random.randint(1, 10, (10, 20)).astype(self.dtype)
        value = 2

        self.inputs = {
            'input': input_np,
            'tensor1': tensor1_np,
            'tensor2': tensor2_np,
        }
        self.attrs = {'value': value}
        self.outputs = {'out': input_np + value * tensor1_np * tensor2_np}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestAddcmulOp_Int64(OpTest):
    """Test int64 dtype - aligned with PyTorch integer type support"""

    def setUp(self):
        self.op_type = "addcmul"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.int64

        input_np = np.random.randint(1, 10, (10, 20)).astype(self.dtype)
        tensor1_np = np.random.randint(1, 10, (10, 20)).astype(self.dtype)
        tensor2_np = np.random.randint(1, 10, (10, 20)).astype(self.dtype)
        value = 3

        self.inputs = {
            'input': input_np,
            'tensor1': tensor1_np,
            'tensor2': tensor2_np,
        }
        self.attrs = {'value': value}
        self.outputs = {'out': input_np + value * tensor1_np * tensor2_np}

    def test_check_output(self):
        self.check_output(check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
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
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestAddcmulBF16Op(OpTest):
    """Test bfloat16 dtype"""

    def setUp(self):
        self.op_type = "addcmul"
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
            input = paddle.static.data(name='input', shape=[4, 4], dtype="bool")
            x3 = paddle.static.data(name='x3', shape=[4, 4], dtype="bool")
            x4 = paddle.static.data(name='x4', shape=[4, 4], dtype="bool")
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
        place = paddle.CPUPlace()
        x = paddle.to_tensor(self.np_input, place=place)
        t1 = paddle.to_tensor(self.np_tensor1, place=place)
        t2 = paddle.to_tensor(self.np_tensor2, place=place)

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
        startup = paddle.static.Program()
        main = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            t1 = paddle.static.data(
                name="tensor1", shape=self.shape, dtype=self.dtype
            )
            t2 = paddle.static.data(
                name="tensor2", shape=self.shape, dtype=self.dtype
            )
            out = paddle.addcmul(x, t1, t2, value=0.5)

            place = paddle.CPUPlace()
            exe = base.Executor(place)
            result = exe.run(
                base.default_main_program(),
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
        place = paddle.CPUPlace()
        input = paddle.to_tensor(self.np_input, place=place)
        tensor1 = paddle.to_tensor(self.np_tensor1, place=place)
        tensor2 = paddle.to_tensor(self.np_tensor2, place=place)
        out = paddle.empty(self.shape, dtype=self.dtype)

        paddle.addcmul(input, tensor1, tensor2, value=0.5, out=out)
        expected = self.np_input + 0.5 * self.np_tensor1 * self.np_tensor2
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_int_dtype(self):
        """Test integer type support (aligned with PyTorch)"""
        paddle.disable_static()
        np_input = np.array([[1, 2], [3, 4]], dtype='int32')
        np_t1 = np.array([[2, 3], [4, 5]], dtype='int32')
        np_t2 = np.array([[1, 1], [2, 2]], dtype='int32')

        x = paddle.to_tensor(np_input)
        t1 = paddle.to_tensor(np_t1)
        t2 = paddle.to_tensor(np_t2)

        out = paddle.addcmul(x, t1, t2, value=2)
        expected = np_input + 2 * np_t1 * np_t2
        np.testing.assert_array_equal(out.numpy(), expected)
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


class TestAddcmulCINNSymbolic(unittest.TestCase):
    """
    Test CINN symbolic shape inference for addcmul operator.
    This covers AddcmulOpInferSymbolicShape in multiary_infer_sym.cc.
    Tests various broadcasting scenarios with dynamic shapes.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CINN requires CUDA")
        if not core.is_compiled_with_cinn():
            self.skipTest("CINN is not compiled")
        paddle.disable_static()
        paddle.seed(2024)

    def _run_with_cinn(self, fn, input_specs, *inputs):
        """Helper to run function with and without CINN and compare results."""
        import paddle.static

        # Run with CINN
        net_cinn = paddle.jit.to_static(
            fn,
            input_spec=input_specs,
            backend="CINN",
            full_graph=True,
        )
        net_cinn.eval()
        cinn_out = net_cinn(*inputs)

        # Run without CINN (dynamic graph reference)
        dy_out = fn(*inputs)

        np.testing.assert_array_equal(cinn_out.numpy(), dy_out.numpy())
        return cinn_out

    def test_cinn_same_shape(self):
        """Test CINN with same shapes (no broadcasting)."""

        def addcmul_fn(x, t1, t2):
            return paddle.addcmul(x, t1, t2, value=0.5)

        x = paddle.randn([8, 16], dtype='float32')
        t1 = paddle.randn([8, 16], dtype='float32')
        t2 = paddle.randn([8, 16], dtype='float32')

        input_specs = [
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
        ]
        self._run_with_cinn(addcmul_fn, input_specs, x, t1, t2)

    def test_cinn_broadcast_tensor2(self):
        """Test CINN with tensor2 broadcasting - covers diffxy > 0 branch."""

        def addcmul_fn(x, t1, t2):
            return paddle.addcmul(x, t1, t2, value=0.5)

        x = paddle.randn([4, 8, 16], dtype='float32')
        t1 = paddle.randn([4, 8, 16], dtype='float32')
        t2 = paddle.randn([8, 16], dtype='float32')  # Will be broadcast

        input_specs = [
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
        ]
        self._run_with_cinn(addcmul_fn, input_specs, x, t1, t2)

    def test_cinn_broadcast_tensor1(self):
        """Test CINN with tensor1 broadcasting - covers diffxy < 0 branch."""

        def addcmul_fn(x, t1, t2):
            return paddle.addcmul(x, t1, t2, value=0.5)

        x = paddle.randn([4, 8, 16], dtype='float32')
        t1 = paddle.randn([8, 16], dtype='float32')  # Will be broadcast
        t2 = paddle.randn([4, 8, 16], dtype='float32')

        input_specs = [
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
        ]
        self._run_with_cinn(addcmul_fn, input_specs, x, t1, t2)

    def test_cinn_broadcast_input(self):
        """Test CINN with input broadcasting - covers diffxyz > 0 branch."""

        def addcmul_fn(x, t1, t2):
            return paddle.addcmul(x, t1, t2, value=0.5)

        x = paddle.randn([8, 16], dtype='float32')  # Will be broadcast
        t1 = paddle.randn([4, 8, 16], dtype='float32')
        t2 = paddle.randn([4, 8, 16], dtype='float32')

        input_specs = [
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
        ]
        self._run_with_cinn(addcmul_fn, input_specs, x, t1, t2)

    def test_cinn_broadcast_z_smaller(self):
        """Test CINN with z (input) dimension smaller - covers diffxyz < 0 branch."""

        def addcmul_fn(x, t1, t2):
            return paddle.addcmul(x, t1, t2, value=0.5)

        x = paddle.randn([4, 8, 16], dtype='float32')
        t1 = paddle.randn([8, 16], dtype='float32')
        t2 = paddle.randn([8, 16], dtype='float32')

        input_specs = [
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
        ]
        self._run_with_cinn(addcmul_fn, input_specs, x, t1, t2)

    def test_cinn_complex_broadcast(self):
        """Test CINN with complex multi-dimensional broadcasting."""

        def addcmul_fn(x, t1, t2):
            return paddle.addcmul(x, t1, t2, value=0.3)

        x = paddle.randn([2, 1, 8, 16], dtype='float32')
        t1 = paddle.randn([1, 4, 1, 16], dtype='float32')
        t2 = paddle.randn([2, 4, 8, 1], dtype='float32')

        input_specs = [
            paddle.static.InputSpec(shape=[None, 1, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1, None, 1, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 8, 1], dtype='float32'),
        ]
        self._run_with_cinn(addcmul_fn, input_specs, x, t1, t2)


class TestAddcmulCINNGrad(unittest.TestCase):
    """
    Test CINN gradient computation for addcmul operator.
    This covers the backward pass with CINN compilation.
    """

    def setUp(self):
        if not core.is_compiled_with_cuda():
            self.skipTest("CINN requires CUDA")
        if not core.is_compiled_with_cinn():
            self.skipTest("CINN is not compiled")
        paddle.disable_static()
        paddle.seed(2024)

    def test_cinn_grad_same_shape(self):
        """Test gradient with CINN for same-shaped tensors."""

        def addcmul_loss(x, t1, t2):
            out = paddle.addcmul(x, t1, t2, value=0.5)
            return out.sum()

        x = paddle.randn([8, 16], dtype='float32')
        t1 = paddle.randn([8, 16], dtype='float32')
        t2 = paddle.randn([8, 16], dtype='float32')
        x.stop_gradient = False
        t1.stop_gradient = False
        t2.stop_gradient = False

        # Dynamic graph reference
        dy_loss = addcmul_loss(x.clone(), t1.clone(), t2.clone())

        # With CINN
        input_specs = [
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
        ]
        net_cinn = paddle.jit.to_static(
            addcmul_loss,
            input_spec=input_specs,
            backend="CINN",
            full_graph=True,
        )

        cinn_loss = net_cinn(x, t1, t2)
        np.testing.assert_array_equal(cinn_loss.numpy(), dy_loss.numpy())

    def test_cinn_grad_broadcast(self):
        """Test gradient with CINN for broadcast tensors."""

        def addcmul_loss(x, t1, t2):
            out = paddle.addcmul(x, t1, t2, value=0.5)
            return out.sum()

        x = paddle.randn([4, 8, 16], dtype='float32')
        t1 = paddle.randn([8, 16], dtype='float32')
        t2 = paddle.randn([4, 8, 16], dtype='float32')
        x.stop_gradient = False
        t1.stop_gradient = False
        t2.stop_gradient = False

        # Dynamic graph reference
        dy_loss = addcmul_loss(x.clone(), t1.clone(), t2.clone())

        # With CINN
        input_specs = [
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 16], dtype='float32'),
        ]
        net_cinn = paddle.jit.to_static(
            addcmul_loss,
            input_spec=input_specs,
            backend="CINN",
            full_graph=True,
        )

        cinn_loss = net_cinn(x, t1, t2)
        np.testing.assert_array_equal(cinn_loss.numpy(), dy_loss.numpy())


# ============================================================
# OpTest broadcast tests for high ranks (backward reduction)
# ============================================================


class TestAddcmulBroadcast4D(OpTest):
    """Rank 4 broadcasting - all three tensors broadcast with reduction"""

    def setUp(self):
        self.op_type = "addcmul"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

        # output: (2,3,10,10)=600, each input >= 100
        input_np = np.random.random((1, 1, 10, 10)).astype(self.dtype)
        tensor1_np = np.random.random((2, 1, 10, 10)).astype(self.dtype)
        tensor2_np = np.random.random((1, 3, 10, 10)).astype(self.dtype)
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


class TestAddcmulBroadcast5D(OpTest):
    """Rank 5 broadcasting - multi-dim reduction"""

    def setUp(self):
        self.op_type = "addcmul"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

        # output: (2,2,5,5,5)=500, each input >= 100
        input_np = np.random.random((1, 2, 5, 5, 5)).astype(self.dtype)
        tensor1_np = np.random.random((2, 1, 5, 5, 5)).astype(self.dtype)
        tensor2_np = np.random.random((2, 2, 1, 5, 5)).astype(self.dtype)
        value = 0.3

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


class TestAddcmulBroadcast6D(OpTest):
    """Rank 6 broadcasting - multi-dim reduction"""

    def setUp(self):
        self.op_type = "addcmul"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

        # output: (2,2,2,5,5,5)=1000, each input >= 100
        input_np = np.random.random((1, 2, 2, 5, 5, 5)).astype(self.dtype)
        tensor1_np = np.random.random((2, 1, 2, 5, 5, 5)).astype(self.dtype)
        tensor2_np = np.random.random((2, 2, 1, 5, 5, 5)).astype(self.dtype)
        value = 0.2

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


class TestAddcmulBroadcastAll3D(OpTest):
    """All three tensors broadcast - exercises all reduction paths"""

    def setUp(self):
        self.op_type = "addcmul"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.dtype = np.float64

        # output: (10,10,10)=1000, each input = 100
        input_np = np.random.random((1, 10, 10)).astype(self.dtype)
        tensor1_np = np.random.random((10, 1, 10)).astype(self.dtype)
        tensor2_np = np.random.random((10, 10, 1)).astype(self.dtype)
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


if __name__ == '__main__':
    unittest.main()
