# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class TestCompatMinMaxBase(unittest.TestCase):
    """The default base class is for testing min-related ops"""

    def __init__(
        self,
        *args,
        test_op=paddle.compat.min,
        origin_op=paddle.min,
        index_op=paddle.argmin,
        test_op_name="paddle.compat.min",
        origin_op_name="paddle.min",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        paddle.disable_static()
        self.test_op = test_op
        self.origin_op = origin_op
        self.index_op = index_op
        self.test_op_name = test_op_name
        self.origin_op_name = origin_op_name
        np.random.seed(1)

    def test_case1_simple_reduce_all(self):
        data = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        val = self.test_op(data)
        if self.test_op_name.endswith("min"):
            self.assertAlmostEqual(val.item(), 1.0)
        else:
            self.assertAlmostEqual(val.item(), 4.0)

    def test_case2_reduce_dim(self):
        """Test dim/keepdim"""
        data = paddle.to_tensor(
            [[[5, 8], [2, 1]], [[7, 3], [9, 6]]], dtype='float32'
        )
        if self.test_op_name.endswith("min"):
            in_dim = 1
            result = self.test_op(data, dim=in_dim)
            expected_res = np.array([[[5, 3], [2, 1]]])
            self.assertEqual(result.values.shape, [2, 2])
            np.testing.assert_array_equal(
                result.values.numpy(), np.array([[2, 1], [7, 3]])
            )
            np.testing.assert_array_equal(
                result.indices.numpy(), np.array([[1, 1], [0, 0]])
            )
        else:
            in_dim = 2
            result = self.test_op(data, dim=in_dim)
            expected_res = np.array([[[7, 8], [9, 6]]])
            self.assertEqual(result.values.shape, [2, 2])
            np.testing.assert_array_equal(
                result.values.numpy(), np.array([[8, 2], [7, 9]])
            )
            np.testing.assert_array_equal(
                result.indices.numpy(), np.array([[1, 0], [0, 0]])
            )

        result_keep = self.test_op(data, dim=0, keepdim=True)
        self.assertEqual(result_keep.values.shape, [1, 2, 2])
        np.testing.assert_array_equal(result_keep.values.numpy(), expected_res)
        result_keep = self.test_op(data, 0, keepdim=True)
        np.testing.assert_array_equal(result_keep.values.numpy(), expected_res)

        result_neg = self.test_op(data, dim=in_dim - 3)
        np.testing.assert_array_equal(
            result_neg.values.numpy(), result.values.numpy()
        )

    def test_case2_grad(self):
        data = paddle.to_tensor(
            [[[1.0, 2.0], [1.0, 3.0]], [[4.0, 1.0], [5.0, 1.0]]],
            dtype='float32',
            stop_gradient=False,
        )
        y = data * 2

        result = self.test_op(y, dim=2)
        result.values.backward()

        if self.test_op_name.endswith("min"):
            expected_grad = np.array(
                [[[2.0, 0.0], [2.0, 0.0]], [[0.0, 2.0], [0.0, 2.0]]]
            )
            expected_grad2 = np.array(
                [[[2.0, 4.0], [0.0, 0.0]], [[8.0, 2.0], [0.0, 0.0]]]
            )
        else:
            expected_grad = np.array(
                [[[0.0, 2.0], [0.0, 2.0]], [[2.0, 0.0], [2.0, 0.0]]]
            )
            expected_grad2 = np.array(
                [[[2.0, 0.0], [0.0, 6.0]], [[0.0, 2.0], [10.0, 0.0]]]
            )
        np.testing.assert_allclose(data.grad.numpy(), expected_grad, atol=1e-6)

        data.clear_grad()
        y = data * data
        result = self.test_op(y, dim=1)
        result[0].backward()
        np.testing.assert_allclose(data.grad.numpy(), expected_grad2, atol=1e-6)

    def test_case3_elementwise(self):
        x = paddle.to_tensor([[1, 5], [4, 2]], dtype='float32')
        y = paddle.to_tensor([[3, 2], [1, 6]], dtype='float32')
        z = paddle.to_tensor([3, 4], dtype='float32')
        broadcast_res = self.test_op(x, z)

        result = self.test_op(x, y)
        if self.test_op_name.endswith("min"):
            np.testing.assert_array_equal(
                result.numpy(), np.array([[1, 2], [1, 2]])
            )
            np.testing.assert_array_equal(
                broadcast_res.numpy(), np.array([[1, 4], [3, 2]])
            )
        else:
            np.testing.assert_array_equal(
                result.numpy(), np.array([[3, 5], [4, 6]])
            )
            np.testing.assert_array_equal(
                broadcast_res.numpy(), np.array([[3, 5], [4, 4]])
            )

    def test_case3_grad(self):
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=paddle.float32, stop_gradient=False
        )
        y = paddle.to_tensor(
            [[0.5, 2.5], [2.0, 3.5]], dtype=paddle.float32, stop_gradient=False
        )

        val = self.test_op(x, y)
        val.backward()

        expected_x_grad = np.array([[0.0, 1.0], [0.0, 0.0]])
        expected_y_grad = np.array([[1.0, 0.0], [1.0, 1.0]])
        if self.test_op_name.endswith("max"):
            expected_x_grad = 1 - expected_x_grad
            expected_y_grad = 1 - expected_y_grad

        np.testing.assert_allclose(x.grad.numpy(), expected_x_grad)
        np.testing.assert_allclose(y.grad.numpy(), expected_y_grad)

    def test_edge_cases(self):
        """Edge cases test"""
        # uniform distributed gradient
        uniform_data = paddle.ones([2, 3], dtype='float64')
        uniform_data.stop_gradient = False
        val = self.test_op(uniform_data)
        val.sum().backward()
        # uniformly distributed
        expected_grad = np.full((2, 3), 1.0 / 6.0)
        np.testing.assert_allclose(uniform_data.grad.numpy(), expected_grad)

        uniform_data.clear_grad()
        val = self.test_op(uniform_data, 0)
        val.values.sum().backward()
        # take_along_axis like gradient behavior
        expected_grad = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(uniform_data.grad.numpy(), expected_grad)

        # 0-dim tensor
        dim0_tensor = paddle.to_tensor(2, dtype='float32')
        val = self.test_op(dim0_tensor)
        np.testing.assert_allclose(val.numpy(), np.array(2.0, dtype=np.float32))

        # 1-dim tensor
        dim1_tensor = paddle.to_tensor([1], dtype='uint8')
        val = self.test_op(dim1_tensor, dim=-1, keepdim=True)
        np.testing.assert_array_equal(
            val[0].numpy(), np.array([1], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            val[1].numpy(), np.array([0], dtype=np.int64)
        )

    def test_compare_with_index_ops_to_origin(self):
        dtypes = ['float32', 'float64', 'int32', 'int64', 'uint8']

        for i, dtype in enumerate(dtypes):
            data = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            # `bfloat16`, `uint8` and `float16` are rejected for min/argmin
            vals_inds = self.test_op(data, dim=0)
            self.assertEqual(vals_inds.values.dtype, data.dtype)
            self.assertEqual(vals_inds.indices.dtype, paddle.int64)

            origin_indices = self.index_op(data, axis=0, dtype="int64")
            if dtype != 'uint8':
                origin_values = self.origin_op(data, axis=0)
            else:
                origin_values = paddle.take_along_axis(
                    data, origin_indices.unsqueeze(0), axis=0
                )
                origin_values.squeeze_(axis=0)
            if i < 4:  # floating point
                np.testing.assert_allclose(
                    vals_inds.values.numpy(), origin_values.numpy()
                )
            else:
                np.testing.assert_array_equal(
                    vals_inds.values.numpy(), origin_values.numpy()
                )
            np.testing.assert_array_equal(
                vals_inds[1].numpy(), origin_indices.numpy()
            )

    def test_case1_out(self):
        data = np.random.randn(4, 5, 6).astype(np.float32)
        x = paddle.to_tensor(data, stop_gradient=False)
        y = paddle.to_tensor(data, stop_gradient=False)
        out = paddle.to_tensor(0)
        self.test_op(x, out=out)
        gt_out = self.origin_op(y)
        gt_out.backward()
        out.backward()

        np.testing.assert_allclose(out.numpy(), gt_out.numpy())
        np.testing.assert_allclose(x.grad.numpy(), y.grad.numpy())

    def test_case2_out(self):
        for type_to_use in (list, tuple):
            data = np.random.randn(3, 17, 5).astype(np.float32)
            x = paddle.to_tensor(data, stop_gradient=False)
            y = paddle.to_tensor(data, stop_gradient=False)
            out = type_to_use((paddle.to_tensor(0), paddle.to_tensor(0)))
            self.test_op(x, dim=1, out=out)
            gt_vals = self.origin_op(y, axis=1)
            gt_inds = self.index_op(y, axis=1)
            gt_vals.backward()
            out[0].backward()

            np.testing.assert_allclose(out[0].numpy(), gt_vals.numpy())
            np.testing.assert_array_equal(out[1].numpy(), gt_inds.numpy())
            np.testing.assert_allclose(x.grad.numpy(), y.grad.numpy())

    def test_case3_out(self):
        data = np.random.randn(3, 4, 5).astype(np.float32)
        x = paddle.to_tensor(data)
        y = paddle.to_tensor(data)
        out = paddle.to_tensor(0)
        self.test_op(x, paddle.ones_like(x), out=out)
        if self.test_op_name.endswith("min"):
            gt_vals = paddle.minimum(x, paddle.ones_like(x))
        else:
            gt_vals = paddle.maximum(x, paddle.ones_like(x))
        np.testing.assert_allclose(out.numpy(), gt_vals.numpy())

    def test_error_handling(self):
        """Test whether correct exception will be thrown. Skip error messages (some of them are long)"""

        err_msg1 = (
            "Tensors with integral type: 'paddle.int32' should stop gradient."
        )
        err_msg2 = (
            f"{self.origin_op_name}() received unexpected keyword arguments 'dim', 'input'. "
            f"\nDid you mean to use {self.test_op_name}() instead?"
        )
        err_msg3 = (
            f"{self.test_op_name}() received unexpected keyword argument 'axis'. "
            f"\nDid you mean to use {self.origin_op_name}() instead?"
        )
        err_msg4 = (
            "Non-CUDA GPU placed Tensor does not have 'paddle.float16' op registered.\n"
            "Paddle support following DataTypes: int32, int64, float64, float32, uint8"
        )
        err_msg5 = (
            "input should be a tensor, but got an instance with type 'list'"
        )

        # empty tensor
        empty_tensor = paddle.to_tensor([], dtype='float32')
        with self.assertRaises(ValueError):
            self.test_op(empty_tensor)

        # mixed parameters case 1
        input_ts = paddle.to_tensor([1, 2, 3], dtype='float32')
        other_ts = paddle.to_tensor([1])
        with self.assertRaises(TypeError):
            self.test_op(input_ts, other=other_ts, dim=0)

        # mixed parameters case 2
        with self.assertRaises(TypeError):
            self.test_op(input_ts, 0, other=other_ts)

        # trying to perform grad ops for integral types
        with self.assertRaises(TypeError) as cm:
            tensor = paddle.ones([2, 2], dtype=paddle.int32)
            tensor.stop_gradient = False
            tensors = self.test_op(tensor, dim=0)
        self.assertEqual(str(cm.exception), err_msg1)

        # explicit None case 1
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, dim=None)

        # explicit None case 2
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, None, keepdim=True)

        # keepdim specified without specifying dim
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, keepdim=True)

        # Wrong *args specification case 1
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, False)

        # Wrong *args specification case 2
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, other_ts, True)

        # Tensor input for dim case 1
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, dim=paddle.to_tensor([0]))

        # Tensor input for dim case 2
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, dim=paddle.to_tensor(0))

        # Tensor input for dim case 3
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, paddle.to_tensor([0]), keepdim=True)

        # Tensor input for dim case 4
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, paddle.to_tensor([0]), True)

        # Duplicate Arguments case 1
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, 0, dim=0)

        # Duplicate Arguments case 2
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, other_ts, other=0)

        # Duplicate Arguments case 3
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, dim=0, other=0, keepdim=True)

        # Wrong API used case 1
        with self.assertRaises(TypeError) as cm:
            self.origin_op(input=input_ts, dim=0)
        self.assertEqual(str(cm.exception), err_msg2)

        # Wrong API used case 2
        with self.assertRaises(TypeError) as cm:
            self.test_op(input_ts, axis=0)
        self.assertEqual(str(cm.exception), err_msg3)

        # Rejected on CPU types
        with self.assertRaises(TypeError) as cm:
            tensor = paddle.to_tensor([1, 2, 3], dtype="float16")
            cpu_tensor = tensor.to("cpu")
            self.test_op(cpu_tensor, dim=0)
        self.assertEqual(str(cm.exception), err_msg4)

        # Wrong input type
        with self.assertRaises(TypeError) as cm:
            self.test_op([1, 2])
        self.assertEqual(str(cm.exception), err_msg5)

        # Wrong second parameter type
        with self.assertRaises(TypeError):
            self.test_op(input_ts, "first_dim")

        paddle.enable_static()
        with (
            self.assertRaises(RuntimeError) as cm,
            paddle.static.program_guard(paddle.static.Program()),
        ):
            x = paddle.static.data(name='x', shape=[None, 6], dtype='float32')
            result0, result1 = self.test_op(
                paddle.zeros([3, 4]),
                dim=1,
                out=(
                    paddle.zeros([3, 4]),
                    paddle.zeros([3, 4], dtype=paddle.int64),
                ),
            )

            place = (
                paddle.CUDAPlace(0)
                if paddle.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            paddle.static.Executor(place).run()
            self.assertEqual(
                str(cm.exception),
                "Using `out` static graph CINN backend is currently not supported. Directly return the tensor tuple instead.\n",
            )
        paddle.disable_static()

        def test_wrong_out_input(dim, out_input):
            with self.assertRaises(TypeError) as cm:
                if dim is None:
                    self.test_op(input_ts, out=out_input)
                else:
                    self.test_op(input_ts, dim=dim, out=out_input)

        test_wrong_out_input(0, [0, paddle.to_tensor(0)])
        test_wrong_out_input(0, paddle.to_tensor(0))
        test_wrong_out_input(None, 0)
        test_wrong_out_input(None, (paddle.to_tensor(0),))

    def _compare_with_origin_static(
        self, input_shape, axis_or_other=0, keepdim=False, use_out=False
    ):
        """Test Case 2 and Case 3 for return output or param output in static graph mode

        TODO(heqianyue): DO NOT set use_out for now!
        Currently, static graph + CINN backend will result in unresolved dependency bug for assign op
        This test is disabled for now, but will be useful when dy2st bug is fixed.
        """
        numel = 1
        for v in input_shape:
            numel *= v
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input_tensor = paddle.arange(numel, dtype=paddle.float32).reshape(
                input_shape
            )

            y = input_tensor**2
            if isinstance(axis_or_other, int):
                if use_out:
                    out = [paddle.to_tensor(0), paddle.to_tensor([0])]
                    self.test_op(y, dim=axis_or_other, keepdim=keepdim, out=out)
                    values, indices = out
                else:
                    values, indices = self.test_op(
                        y, dim=axis_or_other, keepdim=keepdim
                    )
                gt_values = self.origin_op(
                    y, axis=axis_or_other, keepdim=keepdim
                )
                gt_indices = self.index_op(
                    y, axis=axis_or_other, keepdim=keepdim
                )
            else:
                if use_out:
                    out = paddle.to_tensor(0)
                    self.test_op(y, axis_or_other, out=out)
                    values, indices = out, paddle.to_tensor(0)
                else:
                    values, indices = self.test_op(y, axis_or_other)
                if self.test_op_name.endswith("min"):
                    gt_values = paddle.minimum(y, axis=axis_or_other, out=None)
                else:
                    gt_values = paddle.maximum(y, axis=axis_or_other)
                gt_indices = paddle.to_tensor(0)

            place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            values_np, indices_np, gt_values_np, gt_indices_np = exe.run(
                fetch_list=[values, indices, gt_values, gt_indices]
            )
            np.testing.assert_allclose(values_np, gt_values_np)
            np.testing.assert_equal(indices_np, gt_indices_np)
        paddle.disable_static()

    @unittest.skipIf(
        not core.is_compiled_with_cuda(),
        "core is not compiled with CUDA, skipping",
    )
    def test_static_graph(self):
        self._compare_with_origin_static([3, 10, 2], 1)
        self._compare_with_origin_static([3, 10, 2], 0, keepdim=True)
        self._compare_with_origin_static([17], 0)

    @unittest.skipIf(
        not core.is_compiled_with_cuda(),
        "core is not compiled with CUDA, skipping",
    )
    def test_static_unary_shape_infer_1(self):
        # min/max with index is a GPU only op, no need for testing if there is no GPU

        @paddle.jit.to_static(full_graph=True)
        def static_func1(x):
            y = paddle.zeros([2, 3, 4])
            return paddle._C_ops.min_with_index(y, x.shape[0], False, False)

        @paddle.jit.to_static(full_graph=True)
        def static_func2(x):
            y = paddle.zeros([2, 3, 4])
            return paddle._C_ops.min_with_index(y, x.shape[0], True, False)

        input_ts1 = paddle.to_tensor([1])
        input_ts2 = paddle.to_tensor([1, 2])
        val1, ind1 = static_func1(input_ts1)
        val2, ind2 = static_func2(input_ts2)

        self.assertEqual(val1.shape, [2, 4])
        self.assertEqual(ind1.shape, [2, 4])
        self.assertEqual(val2.shape, [2, 3, 1])
        self.assertEqual(ind2.shape, [2, 3, 1])

    @unittest.skipIf(
        not core.is_compiled_with_cuda(),
        "core is not compiled with CUDA, skipping",
    )
    def test_static_unary_shape_infer_2(self):
        # min/max with index is a GPU only op, no need for testing if there is no GPU

        @paddle.jit.to_static(full_graph=True)
        def static_func1(x):
            dim = paddle.arange(0, 1).shape[0]
            y = paddle.zeros([2, 3, 4])
            return paddle._C_ops.max_with_index(y, dim, False, True)

        @paddle.jit.to_static(full_graph=True)
        def static_func2(x):
            dim = paddle.arange(0, 2).shape[0]
            y = paddle.zeros([2, 3, 4])
            return paddle._C_ops.max_with_index(y, dim, True, True)

        x1 = paddle.to_tensor([1])
        x2 = paddle.to_tensor([1, 2])
        val1, ind1 = static_func1(x1)
        val2, ind2 = static_func2(x2)

        self.assertEqual(val1.shape, [])
        self.assertEqual(ind1.shape, [])
        self.assertEqual(val2.shape, [1, 1, 1])
        self.assertEqual(ind2.shape, [1, 1, 1])


class TestCompatMax(TestCompatMinMaxBase):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            test_op=paddle.compat.max,
            origin_op=paddle.max,
            index_op=paddle.argmax,
            test_op_name="paddle.compat.max",
            origin_op_name="paddle.max",
            **kwargs,
        )


if __name__ == '__main__':
    unittest.main()
