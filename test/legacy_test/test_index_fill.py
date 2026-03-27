#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from itertools import combinations

import numpy as np

import paddle
from paddle.base import Program

paddle.enable_static()


def get_all_devices():
    """Return both cpu and gpu devices for coverage."""
    devices = ['cpu']
    if paddle.is_compiled_with_cuda():
        devices.append('gpu')
    return devices


def compute_index_fill_ref(x, axis, index, value):
    perm = list(range(len(x.shape)))
    perm[0] = axis
    perm[axis] = 0

    out = np.transpose(x, perm)
    out[index] = value
    out = np.transpose(out, perm)
    return out


def compute_index_fill_grad_ref(x_shape, axis, index):
    grad = np.ones(x_shape, dtype='float64')
    perm = list(range(len(x_shape)))
    perm[0] = axis
    perm[axis] = 0
    grad = np.transpose(grad, perm)
    grad[index] = 0
    grad = np.transpose(grad, perm)
    return grad


class TestIndexFillAPIBase(unittest.TestCase):
    def setUp(self):
        self.init_setting()
        self.modify_setting()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.index_np = np.array(self.combs[np.random.randint(0, 252)]).astype(
            self.index_type
        )

        self.place = get_all_devices()
        if self.dtype_np == 'float16' and 'cpu' in self.place:
            self.place.remove('cpu')

    def init_setting(self):
        self.dtype_np = 'float64'
        self.index_type = 'int64'
        self.x_shape = (20, 40)
        self.index_size = (5,)
        self.axis = 0
        self.value = -1
        self.combs = list(combinations(list(range(10)), self.index_size[0]))

    def modify_setting(self):
        pass

    def test_static_graph(self):
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(Program()):
                x = paddle.static.data(
                    name="x", shape=self.x_shape, dtype=self.dtype_np
                )
                index = paddle.static.data(
                    name="index", shape=self.index_size, dtype=self.index_type
                )
                out = paddle.index_fill(x, index, self.axis, self.value)
                exe = paddle.static.Executor(place=place)
                feed_list = {"x": self.x_np, "index": self.index_np}
                pd_res = exe.run(
                    paddle.static.default_main_program(),
                    feed=feed_list,
                    fetch_list=[out],
                )[0]
                ref_res = compute_index_fill_ref(
                    self.x_np, self.axis, self.index_np, self.value
                )
                np.testing.assert_allclose(ref_res, pd_res)

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_pd = paddle.to_tensor(self.x_np)
            index_pd = paddle.to_tensor(self.index_np)
            pd_res = paddle.index_fill(x_pd, index_pd, self.axis, self.value)
            ref_res = compute_index_fill_ref(
                self.x_np, self.axis, self.index_np, self.value
            )
            np.testing.assert_allclose(ref_res, pd_res)

    def test_errors(self):
        data_np = np.random.random((10, 10)).astype(np.float32)
        index = paddle.to_tensor([0, 2])

        def test_index_not_tensor():
            res = paddle.index_fill(data_np, [0, 2], axis=-1, value=-1)

        self.assertRaises(ValueError, test_index_not_tensor)

        def test_value_shape():
            res = paddle.index_fill(
                data_np, index, axis=-1, value=paddle.to_tensor([-1, -4])
            )

        self.assertRaises(ValueError, test_value_shape)

        def test_axis_range():
            res = paddle.index_fill(data_np, index, axis=4, value=-1)

        self.assertRaises(ValueError, test_axis_range)


class TestIndexFillAPI1(TestIndexFillAPIBase):
    def modify_setting(self):
        self.dtype_np = 'int64'
        self.index_type = 'int32'
        self.x_shape = (10, 15, 10)
        self.axis = 1


class TestIndexFillAPI2(TestIndexFillAPIBase):
    def modify_setting(self):
        self.dtype_np = 'bool'
        self.index_type = 'int32'
        self.x_shape = (10, 15, 10)
        self.axis = 1
        self.value = True


class TestIndexFillAPI3(TestIndexFillAPIBase):
    def modify_setting(self):
        self.dtype_np = 'float16'
        self.x_shape = (10, 15, 10)
        self.axis = 1
        self.value = 0.5


class TestIndexFillAPINegativeAxis(TestIndexFillAPIBase):
    """Test negative axis to cover `dim < 0` branch in kernel."""

    def modify_setting(self):
        self.dtype_np = 'float64'
        self.x_shape = (10, 15, 10)
        self.axis = -1
        self.value = -2.0


class TestIndexFillAPINegativeAxis2(TestIndexFillAPIBase):
    """Test negative axis=-2 on 3D tensor."""

    def modify_setting(self):
        self.dtype_np = 'float32'
        self.x_shape = (10, 15, 10)
        self.axis = -2
        self.value = 3.0


class TestIndexFillAPI_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.init_setting()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.index_np = np.random.random(self.index_size).astype(
            self.index_type
        )

        self.place = get_all_devices()
        if self.dtype_np == 'float16' and 'cpu' in self.place:
            self.place.remove('cpu')

    def init_setting(self):
        self.dtype_np = 'float64'
        self.index_type = 'int64'
        self.x_shape = (20, 40)
        # test index with zero size
        self.index_size = (0,)
        self.axis = 0
        self.value = -1

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_pd = paddle.to_tensor(self.x_np)
            x_pd.stop_gradient = False
            index_pd = paddle.to_tensor(self.index_np)
            pd_res = paddle.index_fill(x_pd, index_pd, self.axis, self.value)
            ref_res = compute_index_fill_ref(
                self.x_np, self.axis, self.index_np, self.value
            )
            np.testing.assert_allclose(ref_res, pd_res)
            pd_res.sum().backward()
            np.testing.assert_allclose(
                x_pd.grad.numpy(), np.ones_like(self.x_np)
            )
        paddle.enable_static()


class TestIndexFillAPI_ZeroSize2(TestIndexFillAPI_ZeroSize):
    def init_setting(self):
        self.dtype_np = 'float64'
        self.index_type = 'int64'
        self.x_shape = (20, 0)
        self.index_size = (2,)
        self.axis = 0
        self.value = -1


class TestIndexFillGradBase(unittest.TestCase):
    """Test backward to cover IndexFillGradKernel on CPU and GPU."""

    def setUp(self):
        self.init_setting()
        self.modify_setting()
        self.place = get_all_devices()

    def init_setting(self):
        self.dtype_np = 'float64'
        self.index_type = 'int64'
        self.x_shape = (6, 8)
        self.index_np = np.array([0, 2, 4], dtype=self.index_type)
        self.axis = 0
        self.value = -1.0

    def modify_setting(self):
        pass

    def test_dygraph_grad(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_np = np.random.random(self.x_shape).astype(self.dtype_np)
            x_pd = paddle.to_tensor(x_np)
            x_pd.stop_gradient = False
            index_pd = paddle.to_tensor(self.index_np)
            out = paddle.index_fill(x_pd, index_pd, self.axis, self.value)
            loss = out.sum()
            loss.backward()

            expected_grad = compute_index_fill_grad_ref(
                self.x_shape, self.axis, self.index_np
            )
            np.testing.assert_allclose(
                x_pd.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5
            )
        paddle.enable_static()

    def test_static_grad(self):
        paddle.enable_static()
        for place in self.place:
            with paddle.static.program_guard(Program()):
                x = paddle.static.data(
                    name="x", shape=self.x_shape, dtype=self.dtype_np
                )
                x.stop_gradient = False
                index = paddle.static.data(
                    name="index",
                    shape=self.index_np.shape,
                    dtype=self.index_type,
                )
                out = paddle.index_fill(x, index, self.axis, self.value)
                loss = paddle.mean(out)
                grads = paddle.static.gradients(loss, [x])

                exe = paddle.static.Executor(place=place)
                x_np = np.random.random(self.x_shape).astype(self.dtype_np)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": x_np, "index": self.index_np},
                    fetch_list=[grads[0]],
                )
                self.assertEqual(res[0].shape, tuple(self.x_shape))


class TestIndexFillGradAxis1(TestIndexFillGradBase):
    """Test grad with axis=1 to cover inner_size > 1 branch."""

    def modify_setting(self):
        self.x_shape = (4, 6, 5)
        self.index_np = np.array([1, 3], dtype=self.index_type)
        self.axis = 1


class TestIndexFillGradAxis2(TestIndexFillGradBase):
    """Test grad with axis=2 (last dim) to cover outer_size > 1 branch."""

    def modify_setting(self):
        self.x_shape = (3, 4, 5)
        self.index_np = np.array([0, 2, 4], dtype=self.index_type)
        self.axis = 2


class TestIndexFillGradNegativeAxis(TestIndexFillGradBase):
    """Test grad with negative axis to cover dim < 0 branch in grad kernel."""

    def modify_setting(self):
        self.x_shape = (4, 6, 5)
        self.index_np = np.array([2, 4], dtype=self.index_type)
        self.axis = -1


class TestIndexFillGradInt32Index(TestIndexFillGradBase):
    """Test grad with int32 index to cover CastToInt64Kernel branch in grad."""

    def modify_setting(self):
        self.index_type = 'int32'
        self.index_np = np.array([1, 3], dtype=self.index_type)


class TestIndexFillGrad4D(TestIndexFillGradBase):
    """Test grad with 4D tensor."""

    def modify_setting(self):
        self.x_shape = (2, 3, 4, 5)
        self.index_np = np.array([0, 2], dtype=self.index_type)
        self.axis = 2


class TestIndexFillGradFloat32(TestIndexFillGradBase):
    """Test grad with float32 dtype."""

    def modify_setting(self):
        self.dtype_np = 'float32'
        self.x_shape = (5, 8)
        self.index_np = np.array([1, 4], dtype=self.index_type)
        self.axis = 0


class TestIndexFillGradEmptyIndex(TestIndexFillGradBase):
    """Test grad with empty index to cover index.numel() == 0 branch in grad."""

    def modify_setting(self):
        self.index_np = np.array([], dtype=self.index_type)

    def test_dygraph_grad(self):
        paddle.disable_static()
        for place in self.place:
            paddle.device.set_device(place)
            x_np = np.random.random(self.x_shape).astype(self.dtype_np)
            x_pd = paddle.to_tensor(x_np)
            x_pd.stop_gradient = False
            index_pd = paddle.to_tensor(self.index_np)
            out = paddle.index_fill(x_pd, index_pd, self.axis, self.value)
            loss = out.sum()
            loss.backward()
            np.testing.assert_allclose(
                x_pd.grad.numpy(),
                np.ones(self.x_shape, dtype=self.dtype_np),
                rtol=1e-5,
                atol=1e-5,
            )
        paddle.enable_static()


class TestIndexFillInplace(unittest.TestCase):
    """Test inplace index_fill_ to cover the inplace kernel path."""

    def test_inplace_dygraph(self):
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((5, 6)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([1, 3], dtype='int64'))
            paddle.index_fill_(x_pd, index_pd, axis=0, value=-1.0)

            expected = x_np.copy()
            expected[[1, 3], :] = -1.0
            np.testing.assert_allclose(x_pd.numpy(), expected)
        paddle.enable_static()

    def test_inplace_int32_index(self):
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((4, 5)).astype('float32')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([0, 2], dtype='int32'))
            paddle.index_fill_(x_pd, index_pd, axis=1, value=0.0)

            expected = x_np.copy()
            expected[:, [0, 2]] = 0.0
            np.testing.assert_allclose(x_pd.numpy(), expected)
        paddle.enable_static()

    def test_inplace_negative_axis(self):
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((3, 4, 5)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([1, 3], dtype='int64'))
            paddle.index_fill_(x_pd, index_pd, axis=-1, value=2.0)

            expected = x_np.copy()
            expected[:, :, [1, 3]] = 2.0
            np.testing.assert_allclose(x_pd.numpy(), expected)
        paddle.enable_static()


class TestIndexFillNegativeIndex(unittest.TestCase):
    """Test negative index values to cover `idx += dim_size` branch in kernel."""

    def test_negative_index_forward(self):
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((5, 6)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([-1, -3], dtype='int64'))
            pd_res = paddle.index_fill(x_pd, index_pd, 0, -1.0)

            expected = x_np.copy()
            expected[[-1, -3], :] = -1.0
            np.testing.assert_allclose(pd_res.numpy(), expected)
        paddle.enable_static()

    def test_negative_index_backward(self):
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((4, 6)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            x_pd.stop_gradient = False
            index_pd = paddle.to_tensor(np.array([-1, -2], dtype='int64'))
            out = paddle.index_fill(x_pd, index_pd, 0, -1.0)
            loss = out.sum()
            loss.backward()

            expected_grad = np.ones_like(x_np)
            expected_grad[[-1, -2], :] = 0
            np.testing.assert_allclose(
                x_pd.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5
            )
        paddle.enable_static()


class TestIndexFillErrorCases(unittest.TestCase):
    """Test error cases to cover PADDLE_ENFORCE branches in InferMeta."""

    def test_invalid_axis_static(self):
        paddle.enable_static()
        with (
            self.assertRaises(ValueError),
            paddle.static.program_guard(Program()),
        ):
            x = paddle.static.data(name="x_err", shape=(5, 6), dtype='float32')
            index = paddle.static.data(
                name="idx_err", shape=(2,), dtype='int64'
            )
            paddle.index_fill(x, index, axis=5, value=-1)

    def test_index_not_1d_static(self):
        paddle.enable_static()
        with (
            self.assertRaises(ValueError),
            paddle.static.program_guard(Program()),
        ):
            x = paddle.static.data(name="x_err2", shape=(5, 6), dtype='float32')
            index = paddle.static.data(
                name="idx_err2", shape=(2, 3), dtype='int64'
            )
            paddle.index_fill(x, index, axis=0, value=-1)


class TestIndexFillPyTorchStyleArgs(unittest.TestCase):
    """Test PyTorch-style argument order to cover decorator_utils branches."""

    def test_pytorch_positional_order(self):
        """Cover: isinstance(args[1], int) branch in index_fill_decorator."""
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((5, 6)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([0, 2], dtype='int64'))
            # PyTorch order: (input, dim, index, value)
            res = paddle.index_fill(x_pd, 0, index_pd, -1.0)
            expected = x_np.copy()
            expected[[0, 2], :] = -1.0
            np.testing.assert_allclose(res.numpy(), expected)
        paddle.enable_static()

    def test_pytorch_keyword_aliases(self):
        """Cover: 'input' and 'dim' keyword alias branches."""
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((4, 5)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([1, 3], dtype='int64'))
            res = paddle.index_fill(
                input=x_pd, dim=0, index=index_pd, value=-2.0
            )
            expected = x_np.copy()
            expected[[1, 3], :] = -2.0
            np.testing.assert_allclose(res.numpy(), expected)
        paddle.enable_static()

    def test_inplace_pytorch_keyword(self):
        """Cover: 'input'/'dim' keyword alias in index_fill_ inplace."""
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((4, 5)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([0, 2], dtype='int64'))
            paddle.index_fill_(input=x_pd, dim=1, index=index_pd, value=0.0)
            expected = x_np.copy()
            expected[:, [0, 2]] = 0.0
            np.testing.assert_allclose(x_pd.numpy(), expected)
        paddle.enable_static()


class TestIndexFillValueBranches(unittest.TestCase):
    """Test value-related branches in _index_fill_impl."""

    def test_value_as_0d_tensor(self):
        """Cover: value is Variable with numel==1, goes through float(value)."""
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((4, 5)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([1, 3], dtype='int64'))
            value_pd = paddle.to_tensor(-1.0)
            res = paddle.index_fill(x_pd, index_pd, axis=0, value=value_pd)
            expected = x_np.copy()
            expected[[1, 3], :] = -1.0
            np.testing.assert_allclose(res.numpy(), expected)
        paddle.enable_static()

    def test_value_non_scalar_tensor_error(self):
        """Cover: value is Variable with len(value.shape) > 0."""
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_pd = paddle.to_tensor(np.random.random((4, 5)).astype('float64'))
            index_pd = paddle.to_tensor(np.array([1], dtype='int64'))
            value_pd = paddle.to_tensor([-1.0, -2.0])
            with self.assertRaises(ValueError):
                paddle.index_fill(x_pd, index_pd, axis=0, value=value_pd)
        paddle.enable_static()

    def test_value_as_int(self):
        """Cover: value is plain int, goes through float(value) non-Variable branch."""
        paddle.disable_static()
        for place in get_all_devices():
            paddle.device.set_device(place)
            x_np = np.random.random((3, 4)).astype('float64')
            x_pd = paddle.to_tensor(x_np)
            index_pd = paddle.to_tensor(np.array([0, 2], dtype='int64'))
            res = paddle.index_fill(x_pd, index_pd, axis=1, value=0)
            expected = x_np.copy()
            expected[:, [0, 2]] = 0
            np.testing.assert_allclose(res.numpy(), expected)
        paddle.enable_static()


class TestIndexFillGradNegativeAxis2(TestIndexFillGradBase):
    """Test grad with negative axis=-2 to cover dim < 0 branch in grad kernel on 3D."""

    def modify_setting(self):
        self.x_shape = (3, 4, 5)
        self.index_np = np.array([1, 2], dtype=self.index_type)
        self.axis = -2


class TestIndexFillGradInt32IndexAxis1(TestIndexFillGradBase):
    """Test grad with int32 index on axis=1."""

    def modify_setting(self):
        self.index_type = 'int32'
        self.x_shape = (3, 6, 4)
        self.index_np = np.array([0, 3, 5], dtype=self.index_type)
        self.axis = 1


if __name__ == "__main__":
    unittest.main()
