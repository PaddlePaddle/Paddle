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
    # Test Python API
    def setUp(self):
        self.place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 10)).astype('float32')
        tensor1_np = np.random.random((5, 10)).astype('float32')
        tensor2_np = np.random.random((5, 10)).astype('float32')
        value = 0.5

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        expected = input_np + value * tensor1_np * tensor2_np

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_dygraph_api_default_value(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 10)).astype('float32')
        tensor1_np = np.random.random((5, 10)).astype('float32')
        tensor2_np = np.random.random((5, 10)).astype('float32')

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        # Test default value (should be 1.0)
        out = paddle.addcmul(input, tensor1, tensor2)
        expected = input_np + 1.0 * tensor1_np * tensor2_np

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_static_api(self):
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
        paddle.enable_static()

    def test_tensor_method(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((5, 10)).astype('float32')
        tensor1_np = np.random.random((5, 10)).astype('float32')
        tensor2_np = np.random.random((5, 10)).astype('float32')
        value = 0.5

        input = paddle.to_tensor(input_np)
        tensor1 = paddle.to_tensor(tensor1_np)
        tensor2 = paddle.to_tensor(tensor2_np)

        # Test Tensor.addcmul method
        out = input.addcmul(tensor1, tensor2, value=value)
        expected = input_np + value * tensor1_np * tensor2_np

        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)
        paddle.enable_static()

    def test_gradient(self):
        paddle.disable_static(self.place)

        input_np = np.random.random((3, 4)).astype('float32')
        tensor1_np = np.random.random((3, 4)).astype('float32')
        tensor2_np = np.random.random((3, 4)).astype('float32')
        value = 0.5

        input = paddle.to_tensor(input_np, stop_gradient=False)
        tensor1 = paddle.to_tensor(tensor1_np, stop_gradient=False)
        tensor2 = paddle.to_tensor(tensor2_np, stop_gradient=False)

        out = paddle.addcmul(input, tensor1, tensor2, value=value)
        loss = out.sum()
        loss.backward()

        # Verify gradients
        # grad(input) = 1
        # grad(tensor1) = value * tensor2
        # grad(tensor2) = value * tensor1

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


class TestAddcmulOp_ZeroSize2(TestAddcmulOp_ZeroSize):
    def setUp(self):
        self.op_type = "addcmul"
        self.prim_op_type = "comp"
        self.python_api = paddle.addcmul
        self.public_python_api = paddle.addcmul
        self.init_dtype_type()

        self.inputs = {
            'input': np.random.random((5, 0)).astype(self.dtype),
            'tensor1': np.random.random((5, 0)).astype(self.dtype),
            'tensor2': np.random.random((5, 0)).astype(self.dtype),
        }
        self.attrs = {'value': 1.0}
        self.outputs = {
            'out': self.inputs['input']
            + self.attrs['value']
            * self.inputs['tensor1']
            * self.inputs['tensor2']
        }


if __name__ == '__main__':
    unittest.main()
