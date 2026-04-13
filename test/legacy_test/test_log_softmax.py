#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
    get_devices,
    is_custom_device,
)

import paddle
import paddle.nn.functional as F
from paddle.base import core

np.random.seed(10)


def ref_log_softmax(x):
    shiftx = x - np.max(x)
    out = shiftx - np.log(np.exp(shiftx).sum())
    return out


def ref_log_softmax_grad(x, axis):
    if axis < 0:
        axis += len(x.shape)
    out = np.apply_along_axis(ref_log_softmax, axis, x)
    axis_dim = x.shape[axis]
    dout = np.full_like(x, fill_value=1.0 / x.size)
    dx = dout - np.exp(out) * dout.copy().sum(axis=axis, keepdims=True).repeat(
        axis_dim, axis=axis
    )
    return dx


class TestLogSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.prim_op_type = "comp"
        self.python_api = F.log_softmax
        self.public_python_api = F.log_softmax
        self.dtype = 'float64'
        self.shape = [2, 3, 4, 5]
        self.axis = -1
        self.set_attrs()

        x = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)
        self.x_grad = ref_log_softmax_grad(x, self.axis)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis}

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'], ['Out'], user_defined_grads=[self.x_grad], check_pir=True
        )


class TestLogSoftmaxOp_ZeroDim(TestLogSoftmaxOp):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.prim_op_type = "comp"
        self.python_api = F.log_softmax
        self.public_python_api = F.log_softmax
        self.dtype = 'float64'

        x = np.random.uniform(0.1, 1.0, []).astype(self.dtype)
        out = np.array(0.0).astype(self.dtype)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': -1}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'], check_pir=True)


class TestLogSoftmaxShape(TestLogSoftmaxOp):
    def set_attrs(self):
        self.shape = [12, 10]


class TestLogSoftmaxAxis(TestLogSoftmaxOp):
    def set_attrs(self):
        self.axis = 1


class TestLogSoftmaxFP16OP(TestLogSoftmaxOp):
    def set_attrs(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=1e-3, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'], max_relative_error=1e-2, check_pir=True)


class TestLogSoftmaxShapeFP16OP(TestLogSoftmaxFP16OP):
    def set_attrs(self):
        self.dtype = np.float16
        self.shape = [12, 10]


class TestLogSoftmaxAxisFP16OP(TestLogSoftmaxFP16OP):
    def set_attrs(self):
        self.dtype = np.float16
        self.axis = 1


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestLogSoftmaxBF16Op(OpTest):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.prim_op_type = "comp"
        self.python_api = F.log_softmax
        self.public_python_api = F.log_softmax
        self.dtype = np.uint16
        self.shape = [2, 3, 4, 5]
        self.axis = -1

        x = np.random.uniform(0.1, 1.0, self.shape).astype(np.float32)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)
        self.x_grad = ref_log_softmax_grad(x, self.axis)

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.attrs = {'axis': self.axis}

    def test_check_output(self):
        place = get_device_place()
        self.check_output_with_place(place, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        place = get_device_place()
        self.check_grad_with_place(
            place,
            ['X'],
            ['Out'],
            user_defined_grads=[self.x_grad],
            check_pir=True,
        )


class TestLogSoftmaxLargeDimFP16OP(TestLogSoftmaxOp):
    def set_attrs(self):
        self.dtype = np.float16
        self.shape = [16, 100000]


class TestNNLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1.0, 1.0, self.x_shape).astype(np.float32)
        self.place = get_device_place()

    def check_api(self, axis=-1):
        ref_out = np.apply_along_axis(ref_log_softmax, axis, self.x)

        logsoftmax = paddle.nn.LogSoftmax(axis)
        # test static api
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=self.x_shape)
            y = logsoftmax(x)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], ref_out, rtol=1e-05)

        # test dygraph api
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = logsoftmax(x)
        np.testing.assert_allclose(y.numpy(), ref_out, rtol=1e-05)
        paddle.enable_static()

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)


class TestNNFunctionalLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = get_device_place()

    def check_api(self, axis=-1, dtype=None):
        x = self.x.copy()
        if dtype is not None:
            x = x.astype(dtype)
        ref_out = np.apply_along_axis(ref_log_softmax, axis, x)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=self.x_shape)
            y = F.log_softmax(x, axis, dtype)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
        np.testing.assert_allclose(out[0], ref_out, rtol=1e-05)

        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = F.log_softmax(x, axis, dtype)
        np.testing.assert_allclose(y.numpy(), ref_out, rtol=1e-05)
        paddle.enable_static()

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)
        self.check_api(-1, 'float64')

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='X1', shape=[100], dtype='int32')
            self.assertRaises(TypeError, F.log_softmax, x)

            x = paddle.static.data(name='X2', shape=[100], dtype='float32')
            self.assertRaises(TypeError, F.log_softmax, x, dtype='int32')


def _check_cuda_memory_20GB():
    if not hasattr(paddle.device.cuda, 'get_device_properties'):
        return False
    gpu_info = paddle.device.get_device_properties(get_devices()[0])
    return gpu_info.total_memory >= 20 * (1024**3)  # 20GB


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not _check_cuda_memory_20GB(),
    "Need CUDA support and at least 20GB GPU memory",
)
class TestLogSoftmaxLargeOp(unittest.TestCase):
    def test_check_run(self):
        x = paddle.randn([4, 4096, 131072 + 2048])  # 8GB+4*4096*2048
        paddle.nn.functional.log_softmax(x, axis=-1)


class TestLogSoftmaxOp_ZeroSize(OpTest):
    def setUp(self):
        self.op_type = 'log_softmax'
        self.python_api = F.log_softmax
        self.public_python_api = F.log_softmax
        self.dtype = 'float64'
        self.shape = [2, 0, 4, 5]
        self.axis = -1
        self.set_attrs()

        x = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
        # shape is same as x, size is 0.
        out = np.random.random(self.shape).astype(self.dtype)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis}

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'], check_pir=True)


class TestLogSoftmaxParamAlias(unittest.TestCase):
    """Test parameter aliases: input=x, dim=axis."""

    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.uniform(0.1, 1.0, [3, 4]).astype('float32')
        self.x3d_np = np.random.uniform(0.1, 1.0, [2, 3, 4]).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def _ref(self, x_np, axis):
        return np.apply_along_axis(ref_log_softmax, axis, x_np)

    # --- `input` alias for `x` ---

    def test_input_alias_keyword(self):
        x = paddle.to_tensor(self.x_np)
        expected = F.log_softmax(x, axis=-1).numpy()
        result = F.log_softmax(input=x, axis=-1).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_input_alias_with_axis(self):
        x = paddle.to_tensor(self.x_np)
        expected = F.log_softmax(x, axis=0).numpy()
        result = F.log_softmax(input=x, axis=0).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    # --- `dim` alias for `axis` ---

    def test_dim_alias_keyword(self):
        x = paddle.to_tensor(self.x_np)
        expected = F.log_softmax(x, axis=1).numpy()
        result = F.log_softmax(x, dim=1).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_dim_alias_negative(self):
        x = paddle.to_tensor(self.x3d_np)
        expected = F.log_softmax(x, axis=-2).numpy()
        result = F.log_softmax(x, dim=-2).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    # --- Both aliases together ---

    def test_both_aliases(self):
        x = paddle.to_tensor(self.x_np)
        expected = F.log_softmax(x, axis=1).numpy()
        result = F.log_softmax(input=x, dim=1).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_both_aliases_with_dtype(self):
        x = paddle.to_tensor(self.x_np)
        expected = F.log_softmax(x, axis=0, dtype='float64').numpy()
        result = F.log_softmax(input=x, dim=0, dtype='float64').numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        self.assertEqual(result.dtype, np.float64)

    # --- 3D inputs ---

    def test_3d_input_alias_dim0(self):
        x = paddle.to_tensor(self.x3d_np)
        expected = F.log_softmax(x, axis=0).numpy()
        result = F.log_softmax(input=x, dim=0).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_3d_input_alias_dim1(self):
        x = paddle.to_tensor(self.x3d_np)
        expected = F.log_softmax(x, axis=1).numpy()
        result = F.log_softmax(input=x, dim=1).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_3d_input_alias_dim_neg1(self):
        x = paddle.to_tensor(self.x3d_np)
        expected = F.log_softmax(x, axis=-1).numpy()
        result = F.log_softmax(input=x, dim=-1).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    # --- float64 input ---

    def test_float64_input_alias(self):
        x_np = self.x_np.astype('float64')
        x = paddle.to_tensor(x_np)
        expected = F.log_softmax(x, axis=1).numpy()
        result = F.log_softmax(input=x, dim=1).numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # --- Conflict error ---

    def test_conflict_x_and_input_raises(self):
        x = paddle.to_tensor(self.x_np)
        with self.assertRaises(ValueError):
            F.log_softmax(x=x, input=x)

    def test_conflict_axis_and_dim_raises(self):
        x = paddle.to_tensor(self.x_np)
        with self.assertRaises(ValueError):
            F.log_softmax(x, axis=0, dim=1)


class TestLogSoftmaxOutParam(unittest.TestCase):
    """Test out parameter for F.log_softmax."""

    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.uniform(0.1, 1.0, [3, 4]).astype('float32')

    def tearDown(self):
        paddle.enable_static()

    def test_out_param(self):
        x = paddle.to_tensor(self.x_np)
        expected = F.log_softmax(x, axis=-1)
        out = paddle.empty_like(x)
        result = F.log_softmax(x, axis=-1, out=out)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-6)

    def test_out_param_with_dim_alias(self):
        x = paddle.to_tensor(self.x_np)
        expected = F.log_softmax(x, axis=0)
        out = paddle.empty_like(x)
        F.log_softmax(x, dim=0, out=out)
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-6)

    def test_out_param_with_dtype(self):
        x = paddle.to_tensor(self.x_np)
        out = paddle.empty([3, 4], dtype='float64')
        F.log_softmax(x, axis=-1, dtype='float64', out=out)
        self.assertEqual(out.dtype, paddle.float64)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
