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


class TestScatterCompatible(unittest.TestCase):
    def test_non_inplace_origin_scatter(self):
        x = paddle.zeros([3, 4])
        index = paddle.arange(0, 2, dtype=paddle.int64)
        updates = paddle.arange(12, dtype=x.dtype).reshape([3, 4])
        x.stop_gradient = False
        updates.stop_gradient = False
        res_out = paddle.to_tensor(0)
        res = paddle.scatter(
            updates=updates, x=x, overwrite=True, index=index, out=res_out
        )
        gt = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(res.numpy(), gt)
        np.testing.assert_allclose(res_out.numpy(), gt)
        res.backward()
        gt_x_grad = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(x.grad.numpy(), gt_x_grad)

    def test_inplace_origin_scatter(self):
        x = paddle.zeros([4, 4])
        index = paddle.to_tensor([0, 1, 3], dtype=paddle.int64)
        updates = paddle.arange(16, dtype=x.dtype).reshape([4, 4])
        x.stop_gradient = False
        updates.stop_gradient = False
        y = x * x + 2 * x - 1
        res = y.scatter_(updates=updates, index=index, overwrite=True)
        gt = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [-1.0, -1.0, -1.0, -1.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(y.numpy(), gt)
        np.testing.assert_allclose(res.numpy(), gt)
        res.backward()
        gt_x_grad = np.zeros([4, 4], dtype=np.float32)
        gt_x_grad[2, :] = 2
        np.testing.assert_allclose(x.grad.numpy(), gt_x_grad)

    def test_put_along_axis_pass(self):
        inputs = paddle.arange(0, 12, dtype=paddle.float64).reshape([3, 4])
        src = paddle.full_like(inputs, -3)
        index = paddle.ones([3, 3], dtype=paddle.int64)
        gt = np.array(
            [
                [0.0, -8.0, 2.0, 3.0],
                [4.0, -4.0, 6.0, 7.0],
                [8.0, 0.0, 10.0, 11.0],
            ],
            dtype=np.float64,
        )

        arg_cases = [
            [
                1,
            ],
            [],
            [1, index],
            [1, index, src, 'add'],
        ]
        kwarg_cases = [
            {'src': src, 'index': index, 'reduce': 'add'},
            {'src': src, 'index': index, 'reduce': 'add', 'dim': 1},
            {'src': src, 'reduce': 'add'},
            {},
        ]
        for args, kwargs in zip(arg_cases, kwarg_cases):
            res1 = paddle.scatter(inputs, *args, **kwargs)
            res2 = inputs.clone().scatter_(*args, **kwargs)
            np.testing.assert_allclose(res1.numpy(), gt)
            np.testing.assert_allclose(res2.numpy(), gt)

    def test_special_cases_put_along_axis_scatter(self):
        # special case: src is scalar and reduce is None
        inputs = paddle.arange(0, 12, dtype=paddle.float64).reshape([3, 4])
        index = paddle.ones([3, 3], dtype=paddle.int64)
        res = paddle.scatter(inputs, src=-3, reduce=None, index=index, dim=1)
        gt = np.array(
            [
                [0.0, -3.0, 2.0, 3.0],
                [4.0, -3.0, 6.0, 7.0],
                [8.0, -3.0, 10.0, 11.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(res.numpy(), gt)
        inputs.scatter_(src=-3, reduce=None, index=index, dim=1)
        np.testing.assert_allclose(inputs.numpy(), gt)

    def test_error_handling_and_special_cases(self):
        inplace_too_few_args = (
            "Too few arguments in the function call: {p1}, {p2}. Expect one of: \n"
            " - (int dim, Tensor index, Tensor src, *, str reduce, Tensor out = None)\n"
            " - (Tensor index, Tensor updates, bool overwrite, str name = None)"
        )
        non_inplace_too_few_args = (
            "Too few arguments in the function call: {p1}, {p2}. Expect one of: \n"
            " - (Tensor input, int dim, Tensor index, Tensor src, *, str reduce, Tensor out = None)\n"
            " - (Tensor x, Tensor index, Tensor updates, bool overwrite, str name = None)"
        )
        conflicting_params = "`value` is useless when `src` is specified. Be careful for conflicting parameters."

        inplace_put_no_src_or_value = (
            "'paddle.Tensor.scatter_' expect one of the following input pattern: \n"
            " - (int dim, Tensor index, Tensor src (alias value), *, str reduce)\n"
            " - (Tensor index, Tensor updates, bool overwrite, str name = None)\n"
            "However, the input pattern does not match, please check."
        )
        non_inplace_put_no_src_or_value = (
            "'paddle.scatter' expect one of the following input pattern: \n"
            " - (Tensor input, int dim, Tensor index, Tensor src (alias value), *, str reduce, Tensor out = None)\n"
            " - (Tensor x, Tensor index, Tensor updates, bool overwrite, str name = None)\n"
            "However, the input pattern does not match, please check."
        )

        inplace_put_index_input_mismatch = (
            "`index` and `input` must have the same number of dimensions!"
        )
        inplace_put_index_src_mismatch = (
            "`index` and `src` must have the same number of dimensions!"
        )
        put_index_shape_out_of_bound_prefix = "Size does not match at dimension"
        put_index_value_out_of_bound_prefix = (
            "one of element of index is out of bounds"
        )
        dtype_error_prefix = (
            "The data type of index should be one of ['int32', 'int64']"
        )

        dummy_input = paddle.arange(0, 12, dtype=paddle.float64).reshape([3, 4])
        dummy_src = paddle.full_like(dummy_input, -3)
        dummy_index = paddle.ones([3, 3], dtype=paddle.int64)
        dummy_dim = 1
        with self.assertRaises(TypeError) as cm:
            dummy_input.scatter_()
        self.assertEqual(
            str(cm.exception), inplace_too_few_args.format(p1=1, p2=0)
        )

        with self.assertRaises(TypeError) as cm:
            paddle.scatter(input=dummy_input)
        self.assertEqual(
            str(cm.exception), non_inplace_too_few_args.format(p1=0, p2=1)
        )

        with self.assertRaises(TypeError) as cm:
            paddle.scatter(
                dummy_input, dummy_dim, dummy_index, dummy_src, value=dummy_src
            )
        self.assertEqual(str(cm.exception), conflicting_params)

        with self.assertRaises(TypeError) as cm:
            dummy_input.scatter_(
                dummy_dim, dummy_index, dummy_src, value=dummy_src
            )
        self.assertEqual(str(cm.exception), conflicting_params)

        with self.assertRaises(TypeError) as cm:
            paddle.scatter(dummy_input, dummy_dim, dummy_index)
        self.assertEqual(str(cm.exception), non_inplace_put_no_src_or_value)

        with self.assertRaises(TypeError) as cm:
            dummy_input.scatter_(dummy_dim, dummy_index)
        self.assertEqual(str(cm.exception), inplace_put_no_src_or_value)

        with self.assertRaises(ValueError) as cm:
            dummy_input.scatter_(
                dummy_dim,
                paddle.zeros([3, 4, 5], dtype=paddle.int64),
                dummy_src,
            )
        self.assertEqual(str(cm.exception), inplace_put_index_input_mismatch)

        with self.assertRaises(ValueError) as cm:
            dummy_input.scatter_(
                dummy_dim,
                dummy_index,
                paddle.zeros([1], dtype=dummy_input.dtype),
            )
        self.assertEqual(str(cm.exception), inplace_put_index_src_mismatch)

        with self.assertRaises(RuntimeError) as cm:
            dummy_input.scatter_(
                dummy_dim, paddle.zeros([3, 7], dtype=paddle.int64), dummy_src
            )
        self.assertEqual(
            str(cm.exception).startswith(put_index_shape_out_of_bound_prefix),
            True,
        )

        with self.assertRaises(RuntimeError) as cm:
            dummy_input.scatter_(
                dummy_dim,
                paddle.full_like(dummy_input, 7).to(paddle.int64),
                dummy_src,
            )
        self.assertEqual(
            str(cm.exception).startswith(put_index_value_out_of_bound_prefix),
            True,
        )

        with self.assertRaises(TypeError) as cm:
            dummy_input.scatter_(
                dummy_dim, paddle.full_like(dummy_input, 2), dummy_src
            )
        self.assertEqual(str(cm.exception).startswith(dtype_error_prefix), True)


if __name__ == '__main__':
    unittest.main()
