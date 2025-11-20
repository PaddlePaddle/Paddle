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
from op_test import get_device_place, is_custom_device

import paddle


class TestCompatUnfold(unittest.TestCase):
    def _compare_with_origin(
        self, input_tensor, kernel_size, dilation, padding, stride
    ):
        unfold_compat = paddle.compat.nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        unfold_origin = paddle.nn.Unfold(
            kernel_sizes=kernel_size,
            dilations=dilation,
            paddings=padding,
            strides=stride,
        )
        expected_res = unfold_origin(input_tensor).numpy()
        np.testing.assert_allclose(
            unfold_compat(input_tensor).numpy(), expected_res
        )

        # test with tensor input
        to_tensor = lambda x: x if isinstance(x, int) else paddle.to_tensor(x)
        kernel_size = to_tensor(kernel_size)
        dilation = to_tensor(dilation)
        padding = to_tensor(padding)
        stride = to_tensor(stride)
        unfold_compat = paddle.compat.nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        np.testing.assert_allclose(
            unfold_compat(input_tensor).numpy(), expected_res
        )

    def test_compare_with_origin(self):
        input_shape = (3, 4, 5, 6)
        input_tensor = paddle.arange(360, dtype=paddle.float32).reshape(
            input_shape
        )
        self._compare_with_origin(input_tensor, [3, 3], [1, 1], (1, 2), [1, 1])

        input_shape = (5, 10, 13, 13)
        input_tensor = paddle.ones(input_shape, dtype=paddle.float64)
        self._compare_with_origin(input_tensor, [4, 4], [2, 2], 1, (1, 2))

        input_shape = (12, 4, 10, 10)
        input_tensor = paddle.ones(input_shape, dtype=paddle.float64)
        self._compare_with_origin(input_tensor, 3, 2, 1, (1, 1))

    def test_error_handling(self):
        """Test whether there will be correct exception when users pass paddle.split kwargs in paddle.compat.split, vice versa."""
        x = paddle.randn([3, 9, 5])

        msg_gt_1 = "paddle.nn.Unfold() received unexpected keyword arguments 'dilation', 'stride'. \nDid you mean to use paddle.compat.nn.Unfold() instead?"
        msg_gt_2 = "paddle.compat.nn.Unfold() received unexpected keyword argument 'paddings'. \nDid you mean to use paddle.nn.Unfold() instead?"
        msg_gt_3 = "The `padding` field of paddle.compat.nn.Unfold can only have size 1 or 2, now len=4. \nDid you mean to use paddle.nn.Unfold() instead?"
        msg_gt_4 = "paddle.compat.nn.Unfold does not allow paddle.Tensor or pir.Value as inputs in static graph mode."

        with self.assertRaises(TypeError) as cm:
            unfold = paddle.nn.Unfold([3, 3], dilation=[2, 2], stride=[1, 1])
        self.assertEqual(str(cm.exception), msg_gt_1)

        with self.assertRaises(TypeError) as cm:
            unfold = paddle.compat.nn.Unfold([3, 3], paddings=[2, 1])
        self.assertEqual(str(cm.exception), msg_gt_2)

        with self.assertRaises(ValueError) as cm:
            unfold = paddle.compat.nn.Unfold([3, 3], padding=[2, 1, 2, 2])
            res = unfold(paddle.ones([2, 2, 5, 5]))
        self.assertEqual(str(cm.exception), msg_gt_3)

        with self.assertRaises(TypeError) as cm:
            paddle.enable_static()
            input_data = np.random.randn(2, 4, 8, 8).astype(np.float32)
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(
                    name='x', shape=[None, None, 8, 8], dtype='float32'
                )
                place = (
                    get_device_place()
                    if (paddle.is_compiled_with_cuda() or is_custom_device())
                    else paddle.CPUPlace()
                )
                unfold_pass = paddle.compat.nn.Unfold(
                    kernel_size=paddle.to_tensor([3, 3]),
                    padding=paddle.to_tensor([1, 2]),
                )
                result = unfold_pass(x)
                exe = paddle.static.Executor(place)
                feed = {'x': input_data}
                exe_res = exe.run(feed=feed)
            paddle.disable_static()
        self.assertEqual(str(cm.exception), msg_gt_4)


if __name__ == '__main__':
    unittest.main()
