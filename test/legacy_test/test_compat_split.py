# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.compat import split


class TestCompatSplit(unittest.TestCase):
    def _compare_with_origin(self, input_tensor, size, axis=0):
        pd_results = split(input_tensor, size, dim=axis)

        if isinstance(size, int):
            shape_on_axis = input_tensor.shape[axis]
            remaining_num = shape_on_axis % size
            num_sections = shape_on_axis // size
            if remaining_num == 0:
                size = num_sections
            else:
                size = [size for _ in range(num_sections)]
                size.append(remaining_num)

        origin_results = paddle.split(
            input_tensor, num_or_sections=size, axis=axis
        )

        self.assertEqual(len(origin_results), len(pd_results))

        # check shape and output section size of the output
        for origin_ts, pd_ts in zip(origin_results, pd_results):
            np.testing.assert_allclose(origin_ts.numpy(), pd_ts.numpy())

    def test_basic_split(self):
        """Test basic splitting with integer size"""
        data = paddle.arange(12).reshape([3, 4]).astype('float32')
        self._compare_with_origin(data, 1, 0)
        self._compare_with_origin(data, 2, 1)

    def test_split_with_list_sections(self):
        """Test splitting with list of section sizes"""
        data = paddle.rand([10, 5])
        self._compare_with_origin(data, [3, 2, 5], 0)
        self._compare_with_origin(data, [1, 4], -1)

    def test_chained_operations(self):
        """Test split with complex operation chain"""
        x = paddle.rand([8, 12])
        y = paddle.sin(x) * 2.0 + paddle.exp(x) / 3.0
        z = paddle.nn.functional.relu(y)

        z1, z2 = split(z, 7, dim=1)

        self.assertEqual(z1.shape, [8, 7])
        self.assertEqual(z2.shape, [8, 5])

        z_np = z.numpy()
        np.testing.assert_allclose(z_np[:, :7], z1.numpy())
        np.testing.assert_allclose(z_np[:, 7:], z2.numpy())

    def test_split_grad(self):
        """Test backprop for split, in1 and in2 are computed by
        compat.split and original split"""

        def get_tensors():
            np.random.seed(114514)
            np_arr = np.random.normal(0, 1, [2, 3, 4, 5])
            return paddle.to_tensor(np_arr), paddle.to_tensor(np_arr)

        in1, in2 = get_tensors()
        in1.stop_gradient = False
        in2.stop_gradient = False

        def computation_graph(in_tensor):
            y = in_tensor * 2.3 + 3.0
            y = paddle.maximum(y, paddle.to_tensor([0], dtype=paddle.float32))
            return y.mean(axis=0)

        out1 = computation_graph(in1)
        out2 = computation_graph(in2)

        packs1 = paddle.compat.split(out1, 2, dim=2)
        packs2 = paddle.split(out2, [2, 2, 1], axis=2)

        res1 = packs1[0] + packs1[1] + packs1[2]
        res2 = packs2[0] + packs2[1] + packs2[2]
        res1.backward()
        res2.backward()
        np.testing.assert_allclose(in1.grad.numpy(), in2.grad.numpy())

    def test_empty_dim(self):
        """Split with empty dim"""
        in_tensor = paddle.arange(72, dtype=paddle.int64).reshape([3, 12, 2])
        self._compare_with_origin(in_tensor, [5, 0, 7], axis=1)

    def test_split_with_one_block(self):
        """Resulting tuple should be of length 1"""
        in_tensor = paddle.arange(60, dtype=paddle.float32).reshape([3, 4, 5])
        self._compare_with_origin(in_tensor, 5, paddle.to_tensor([-1]))
        self._compare_with_origin(in_tensor, [5], paddle.to_tensor(2))

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        x = paddle.arange(5)
        s1, s2 = split(x, [3, 2])
        np.testing.assert_allclose(s1.numpy(), [0, 1, 2])
        np.testing.assert_allclose(s2.numpy(), [3, 4])

        x = paddle.rand([2, 2, 2])
        a, b = split(x, 1, 2)
        self.assertEqual(a.shape, [2, 2, 1])

        # invalid split sections
        with self.assertRaises(ValueError):
            split(x, [3, 1], 1)

        # invalid split axis
        with self.assertRaises(ValueError):
            split(x, 2, 3)

    def test_error_hint(self):
        """Test whether there will be correct exception when users pass paddle.split kwargs in paddle.compat.split, vice versa."""
        x = paddle.randn([3, 9, 5])

        msg_gt_1 = (
            "paddle.split() received unexpected keyword arguments 'dim', 'split_size_or_sections', 'tensor'. "
            "\nDid you mean to use paddle.compat.split() instead?"
        )
        msg_gt_2 = (
            "paddle.compat.split() received unexpected keyword argument 'num_or_sections'. "
            "\nDid you mean to use paddle.split() instead?"
        )
        msg_gt_3 = "(InvalidArgument) The dim is expected to be in range of [-3, 3), but got 3"
        msg_gt_4 = "paddle.compat.split expects split_sizes have only non-negative entries, but got size = -5 on dim 2"

        split_size = paddle.to_tensor([3])
        msg_gt_5 = (
            "The type of 'split_size_or_sections' in split must be int, list or tuple in imperative mode, but "
            f"received {type(split_size)}."
        )

        with self.assertRaises(TypeError) as cm:
            tensors = paddle.split(tensor=x, split_size_or_sections=3, dim=0)
        self.assertEqual(str(cm.exception), msg_gt_1)

        with self.assertRaises(TypeError) as cm:
            tensors = split(x, num_or_sections=3, dim=0)
        self.assertEqual(str(cm.exception), msg_gt_2)

        with self.assertRaises(ValueError) as cm:
            tensors = split(x, 3, dim=3)
        self.assertEqual(str(cm.exception), msg_gt_3)

        with self.assertRaises(ValueError) as cm:
            tensors = split(x, [3, 3, -5], -2)
        self.assertEqual(str(cm.exception), msg_gt_4)

        with self.assertRaises(TypeError) as cm:
            tensors = split(x, split_size, 1)
        self.assertEqual(str(cm.exception), msg_gt_5)


class TestFunctionalSplit(unittest.TestCase):
    def test_functional_split(self):
        x = paddle.rand([3, 9, 5])
        out_expect = paddle.compat.split(
            x, split_size_or_sections=[2, 3, 4], dim=1
        )
        out_res = paddle.functional.split(
            x, split_size_or_sections=[2, 3, 4], dim=1
        )
        for expect, res in zip(out_expect, out_res):
            np.testing.assert_allclose(
                expect.numpy(), res.numpy(), atol=1e-8, rtol=1e-8
            )

        out_expect = paddle.compat.split(x, split_size_or_sections=3, dim=-2)
        out_res = paddle.functional.split(x, split_size_or_sections=3, dim=-2)
        for expect, res in zip(out_expect, out_res):
            np.testing.assert_allclose(
                expect.numpy(), res.numpy(), atol=1e-8, rtol=1e-8
            )


if __name__ == '__main__':
    unittest.main()
