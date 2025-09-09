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


class TestGatherCompatible(unittest.TestCase):
    def test_non_inplace_origin_gather(self):
        x = paddle.arange(12, dtype=paddle.float32).reshape([3, 4])
        index = paddle.to_tensor([0, 1, 1], dtype=paddle.int64)
        x.stop_gradient = False
        res_out = paddle.to_tensor(0)
        res = paddle.gather(x, axis=1, index=index, out=res_out)
        gt = np.array(
            [[0.0, 1.0, 1.0], [4.0, 5.0, 5.0], [8.0, 9.0, 9.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(res.numpy(), gt)
        np.testing.assert_allclose(res_out.numpy(), gt)
        res.backward()
        gt_x_grad = np.array(
            [[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(x.grad.numpy(), gt_x_grad)

    def test_take_along_axis_pass(self):
        inputs = paddle.arange(0, 12, dtype=paddle.float64).reshape([3, 4])
        index = paddle.ones([2, 4], dtype=paddle.int64)
        gt = np.array(
            [[1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]],
            dtype=np.float64,
        )

        arg_cases = [
            [1],
            [],
            [1, index],
        ]
        kwarg_cases = [
            {
                'index': index,
            },
            {'index': index, 'dim': 1},
            {},
        ]
        for args, kwargs in zip(arg_cases, kwarg_cases):
            res = paddle.gather(inputs, *args, **kwargs)
            np.testing.assert_allclose(res.numpy(), gt)

    def test_error_handling_and_special_cases(self):
        too_few_args = (
            "Too few arguments in the function call: {p1}, {p2}. Expect one of: \n"
            " - (Tensor input, int dim, Tensor index, *, Tensor out = None)\n"
            " - (Tensor x, Tensor index, int axis, str name = None, Tensor out = None)"
        )

        dummy_input = paddle.arange(0, 12, dtype=paddle.float64).reshape([3, 4])
        dummy_index = paddle.ones([3, 3], dtype=paddle.int64)
        dummy_dim = 1
        with self.assertRaises(TypeError) as cm:
            paddle.gather(dummy_input)
        self.assertEqual(str(cm.exception), too_few_args.format(p1=1, p2=0))

        with self.assertRaises(TypeError) as cm:
            paddle.gather(input=dummy_input)
        self.assertEqual(str(cm.exception), too_few_args.format(p1=0, p2=1))


if __name__ == '__main__':
    paddle.set_device('cpu')
    unittest.main()
