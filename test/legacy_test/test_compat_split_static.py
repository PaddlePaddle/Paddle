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


class TestCompatSplitStatic(unittest.TestCase):
    def _compare_with_origin_static(
        self, input_shape, size, axis=0, dim_rank=-1
    ):
        """size_dim: -1 means we input size by int, 0 means 0-size tensor, 1 means tensor with shape [1]"""
        numel = 1
        for v in input_shape:
            numel *= v
        input_axis = axis
        if dim_rank == 0:
            input_axis = paddle.to_tensor(axis)
        elif dim_rank == 1:
            input_axis = paddle.to_tensor([axis])
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input_tensor = paddle.arange(numel, dtype=paddle.float32).reshape(
                input_shape
            )
            pd_results = split(input_tensor, size, dim=input_axis)

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
            assert len(pd_results) == len(origin_results), "length mismatched"
            place = (
                paddle.CUDAPlace(0)
                if paddle.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            results = exe.run(fetch_list=[*origin_results, *pd_results])
            length_needed = len(results) // 2
            for i in range(length_needed):
                np.testing.assert_allclose(
                    results[i], results[i + length_needed]
                )
        paddle.disable_static()

    def test_split_composite_static(self):
        paddle.seed(114514)

        def get_tensors():
            np.random.seed(114514)
            np_arr = np.random.normal(0, 1, [2, 3, 4, 5])
            return paddle.to_tensor(np_arr), paddle.to_tensor(np_arr)

        in1, in2 = get_tensors()
        in1.stop_gradient = False
        in2.stop_gradient = False

        @paddle.jit.to_static
        def computation_graph(in1: paddle.Tensor, in2: paddle.Tensor):
            y1 = in1 * 1.5 + 1.0
            y1 = paddle.minimum(y1, paddle.to_tensor([0], dtype=paddle.float32))
            out1 = y1.mean(axis=0)

            y2 = in2 * 1.5 + 1.0
            y2 = paddle.minimum(y2, paddle.to_tensor([0], dtype=paddle.float32))
            out2 = y2.mean(axis=0)

            packs1 = paddle.compat.split(out1, 2, dim=2)
            packs2 = paddle.split(out2, [2, 2, 1], axis=2)

            res1 = packs1[0] + packs1[1] + packs1[2]
            res2 = packs2[0] + packs2[1] + packs2[2]

            return res1, res2

        res1, res2 = computation_graph(in1, in2)
        np.testing.assert_allclose(res1.numpy(), res2.numpy())

    def test_static_graph(self):
        """Test static graph execution"""
        # fixed random seed for reproducibility
        np.random.seed(114514)
        # old static graph mode
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[None, 6], dtype='float32')
            result0, result1 = split(x, split_size_or_sections=[3, 3], dim=1)
            output = result0 * 2.0 + paddle.sin(result1)

            place = (
                paddle.CUDAPlace(0)
                if paddle.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)

            input_data = np.random.rand(3, 6).astype('float32')
            feed = {'x': input_data}

            results = exe.run(feed=feed, fetch_list=[result0, result1, output])

            pd_result0, pd_result1 = results[0], results[1]
            np.testing.assert_allclose(input_data[:, :3], pd_result0)
            np.testing.assert_allclose(input_data[:, 3:], pd_result1)

            expected_output = input_data[:, :3] * 2.0 + np.sin(
                input_data[:, 3:]
            )
            np.testing.assert_allclose(
                expected_output, results[2], rtol=1e-4, atol=1e-4
            )

        paddle.disable_static()

    def test_error_hint(self):
        """Test whether there will be correct exception when users pass paddle.split kwargs in paddle.compat.split, vice versa."""

        msg_gt_1 = "split_size_or_sections must be greater than 0."
        msg_gt_2 = "len(split_size_or_sections) must not be more than input.shape[dim]."
        msg_gt_3 = "The type of 'split_size_or_sections' in split must be int, list or tuple in imperative mode."
        msg_gt_4 = (
            "'dim' is not allowed to be a pir.Value in a static graph: "
            "\npir.Value can not be used for indexing python lists/tuples."
        )

        paddle.enable_static()
        with self.assertRaises(AssertionError) as cm:
            x = paddle.randn([3, 4, 5])
            tensors = split(x, -2, dim=0)
        self.assertEqual(str(cm.exception), msg_gt_1)

        with self.assertRaises(AssertionError) as cm:
            x = paddle.randn([3, 4, 5])
            tensors = split(x, (1, 1, 1, 1, 2, 2), dim=-1)
        self.assertEqual(str(cm.exception), msg_gt_2)

        with self.assertRaises(TypeError) as cm:
            x = paddle.randn([3, 4, 5])
            tensors = split(x, paddle.to_tensor(2), dim=2)
        self.assertEqual(str(cm.exception), msg_gt_3)

        with self.assertRaises(TypeError) as cm:
            x = paddle.randn([3, 4, 5])
            tensors = split(x, 2, dim=paddle.to_tensor(2))
        paddle.disable_static()
        self.assertEqual(str(cm.exception), msg_gt_4)

    def test_basic_split(self):
        """Test basic splitting with integer size"""
        input_shape = [3, 6]
        self._compare_with_origin_static(input_shape, 1, 0)
        self._compare_with_origin_static(input_shape, 3, -1)
        self._compare_with_origin_static(input_shape, 4, dim_rank=0)
        self._compare_with_origin_static(input_shape, 3, dim_rank=1)


if __name__ == '__main__':
    unittest.main()
