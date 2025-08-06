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
                expected_output, results[2], rtol=1e-3, atol=1e-3
            )

        paddle.disable_static()

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

        msg_gt_1 = "split() received unexpected keyword arguments 'tensor', 'split_size_or_sections', 'dim'. \nDid you mean to use paddle.compat.split() instead?"
        msg_gt_2 = "split() received unexpected keyword argument 'num_or_sections'. \nDid you mean to use paddle.split() instead?"

        with self.assertRaises(TypeError) as cm:
            tensors = paddle.split(tensor=x, split_size_or_sections=3, dim=0)
        self.assertEqual(str(cm.exception), msg_gt_1)

        with self.assertRaises(TypeError) as cm:
            tensors = split(x, num_or_sections=3, dim=0)
        self.assertEqual(str(cm.exception), msg_gt_2)


if __name__ == '__main__':
    unittest.main()
