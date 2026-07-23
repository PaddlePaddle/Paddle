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

import paddle
import paddle.compat.nn.functional as F_compat


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

    def test_public_attributes_drive_forward(self):
        layer = paddle.compat.nn.Unfold(1, dilation=1, padding=0, stride=1)
        self.assertEqual(layer.kernel_size, 1)
        self.assertEqual(layer.dilation, 1)
        self.assertEqual(layer.padding, 0)
        self.assertEqual(layer.stride, 1)
        self.assertIn("kernel_size=1", repr(layer))

        layer.kernel_size = 2
        x = paddle.arange(12, dtype="float32").reshape([1, 3, 4])
        expected = F_compat.unfold(x, kernel_size=2)
        np.testing.assert_array_equal(layer(x).numpy(), expected.numpy())

    def test_error_handling(self):
        """Test whether there will be correct exception when users pass paddle.split kwargs in paddle.compat.split, vice versa."""
        x = paddle.randn([3, 9, 5])

        msg_gt_1 = "paddle.nn.Unfold() received unexpected keyword arguments 'dilation', 'stride'. \nDid you mean to use paddle.compat.nn.Unfold() instead?"
        msg_gt_2 = "paddle.compat.nn.Unfold() received unexpected keyword argument 'paddings'. \nDid you mean to use paddle.nn.Unfold() instead?"

        with self.assertRaises(TypeError) as cm:
            unfold = paddle.nn.Unfold([3, 3], dilation=[2, 2], stride=[1, 1])
        self.assertEqual(str(cm.exception), msg_gt_1)

        with self.assertRaises(TypeError) as cm:
            unfold = paddle.compat.nn.Unfold([3, 3], paddings=[2, 1])
        self.assertEqual(str(cm.exception), msg_gt_2)


class TestCompatFunctionalUnfold(unittest.TestCase):
    def _compare_with_origin(
        self, input_tensor, kernel_size, dilation, padding, stride
    ):
        out_compat = F_compat.unfold(
            input=input_tensor,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        out_origin = paddle.nn.functional.unfold(
            x=input_tensor,
            kernel_sizes=kernel_size
            if not isinstance(kernel_size, paddle.Tensor)
            else kernel_size.tolist(),
            dilations=dilation
            if not isinstance(dilation, paddle.Tensor)
            else dilation.tolist(),
            paddings=padding
            if not isinstance(padding, paddle.Tensor)
            else padding.tolist(),
            strides=stride
            if not isinstance(stride, paddle.Tensor)
            else stride.tolist(),
        )

        expected_res = out_origin.numpy()
        np.testing.assert_allclose(out_compat.numpy(), expected_res)

        to_tensor = lambda x: x if isinstance(x, int) else paddle.to_tensor(x)
        k_t = (
            to_tensor(kernel_size)
            if not isinstance(kernel_size, paddle.Tensor)
            else kernel_size
        )
        d_t = (
            to_tensor(dilation)
            if not isinstance(dilation, paddle.Tensor)
            else dilation
        )
        p_t = (
            to_tensor(padding)
            if not isinstance(padding, paddle.Tensor)
            else padding
        )
        s_t = (
            to_tensor(stride)
            if not isinstance(stride, paddle.Tensor)
            else stride
        )

        out_compat_tensor = F_compat.unfold(
            input=input_tensor,
            kernel_size=k_t,
            dilation=d_t,
            padding=p_t,
            stride=s_t,
        )
        np.testing.assert_allclose(out_compat_tensor.numpy(), expected_res)

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

    def test_unbatched_cpu_dtypes(self):
        base = np.arange(12).reshape([1, 3, 4])

        def expected_unfold(array):
            return np.stack(
                [
                    array[:, i : i + 2, j : j + 2].reshape(-1)
                    for i in range(2)
                    for j in range(3)
                ],
                axis=-1,
            )

        cases = (
            (paddle.float16, base.astype(np.float32)),
            (paddle.bfloat16, base.astype(np.float32)),
            (paddle.bool, (base % 2).astype(bool)),
            (
                paddle.complex64,
                base.astype(np.complex64)
                + 1j * (base + 1).astype(np.complex64),
            ),
            (
                paddle.complex128,
                base.astype(np.complex128)
                + 1j * (base + 1).astype(np.complex128),
            ),
        )
        for dtype, input_np in cases:
            x = paddle.to_tensor(input_np, dtype=dtype, place=paddle.CPUPlace())
            out = F_compat.unfold(x, kernel_size=2)
            self.assertEqual(out.dtype, dtype)
            if dtype == paddle.bfloat16:
                actual = out.astype("float32").numpy()
            else:
                actual = out.numpy()
            np.testing.assert_allclose(actual, expected_unfold(input_np))

    def test_error_handling(self):
        """Test whether there will be correct exception when users pass incorrect kwargs."""
        x = paddle.randn([3, 9, 5, 5])

        msg_gt_wrong_key = "paddle.compat.nn.functional.unfold() received unexpected keyword argument 'paddings'. \nDid you mean to use paddle.nn.functional.unfold() instead?"

        with self.assertRaises(TypeError) as cm:
            F_compat.unfold(x, [3, 3], paddings=[2, 1])
        self.assertEqual(str(cm.exception), msg_gt_wrong_key)

        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
