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
import paddle.compat.nn.functional as F


class TestCompatPad(unittest.TestCase):
    def test_basic_pad(self):
        """Test basic splitting with integer size"""
        gt = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [0.0, 0.0]],
                [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        x_shape = (3, 3, 2)
        x = (
            paddle.arange(
                paddle.prod(paddle.Tensor(x_shape)), dtype=paddle.float32
            ).reshape(x_shape)
            + 1
        )
        result = F.pad(
            input=x, pad=[0, 0, 0, 1, 2, 3], mode='constant', value=0
        )

        np.testing.assert_allclose(result.numpy(), gt)

    def test_constant_fast_pass(self):
        gt_res = np.array(
            [
                [[-1, -1, -1, -1, -1], [-1, 0, 1, -1, -1], [-1, 2, 3, -1, -1]],
                [[-1, -1, -1, -1, -1], [-1, 4, 5, -1, -1], [-1, 6, 7, -1, -1]],
                [
                    [-1, -1, -1, -1, -1],
                    [-1, 8, 9, -1, -1],
                    [-1, 10, 11, -1, -1],
                ],
            ],
            dtype=np.int64,
        )

        def const_pad_dy(x, pad_shape):
            return F.pad(input=x, pad=pad_shape, mode='constant', value=-1)

        @paddle.jit.to_static(full_graph=True)
        def const_pad_st(x, pad_shape):
            return F.pad(
                input=x,
                pad=pad_shape,
                mode='constant',
                value=paddle.to_tensor(-1),
            )

        x = paddle.arange(12).reshape(3, 2, 2)
        res_dy = const_pad_dy(x, [1, 2, 1])
        res_st = const_pad_st(x, [1, 2, 1])

        np.testing.assert_array_equal(res_dy.numpy(), gt_res)
        np.testing.assert_array_equal(res_st.numpy(), gt_res)

    def test_single_dim(self):
        gt = np.array([0, 0, 1, 2], dtype=np.float64)
        x_shape = 2
        x = paddle.arange(2, dtype=paddle.float64) + 1
        result = F.pad(x, mode='constant', pad=[2])
        np.testing.assert_allclose(result.numpy(), gt)

    def test_no_pad(self):
        gt = np.array(
            [
                [
                    [
                        [[0.0, 0.0, 1.0], [2.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
                        [[4.0, 4.0, 5.0], [6.0, 6.0, 7.0], [6.0, 6.0, 7.0]],
                    ],
                    [
                        [
                            [8.0, 8.0, 9.0],
                            [10.0, 10.0, 11.0],
                            [10.0, 10.0, 11.0],
                        ],
                        [
                            [12.0, 12.0, 13.0],
                            [14.0, 14.0, 15.0],
                            [14.0, 14.0, 15.0],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [16.0, 16.0, 17.0],
                            [18.0, 18.0, 19.0],
                            [18.0, 18.0, 19.0],
                        ],
                        [
                            [20.0, 20.0, 21.0],
                            [22.0, 22.0, 23.0],
                            [22.0, 22.0, 23.0],
                        ],
                    ],
                    [
                        [
                            [24.0, 24.0, 25.0],
                            [26.0, 26.0, 27.0],
                            [26.0, 26.0, 27.0],
                        ],
                        [
                            [28.0, 28.0, 29.0],
                            [30.0, 30.0, 31.0],
                            [30.0, 30.0, 31.0],
                        ],
                    ],
                ],
            ],
            dtype=np.float64,
        )
        x = paddle.arange(32, dtype=paddle.float64).reshape([2] * 5)
        result = F.pad(x, mode='replicate', pad=[1, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(result.numpy(), gt)

    def test_static_graph_circular(self):
        cir_gt = np.array(
            [
                [
                    [10.0, 11.0, 8.0, 9.0, 10.0, 11.0, 8.0],
                    [2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0],
                    [6.0, 7.0, 4.0, 5.0, 6.0, 7.0, 4.0],
                    [10.0, 11.0, 8.0, 9.0, 10.0, 11.0, 8.0],
                ],
                [
                    [22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0],
                    [14.0, 15.0, 12.0, 13.0, 14.0, 15.0, 12.0],
                    [18.0, 19.0, 16.0, 17.0, 18.0, 19.0, 16.0],
                    [22.0, 23.0, 20.0, 21.0, 22.0, 23.0, 20.0],
                ],
            ],
            dtype=np.float32,
        )
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input_tensor = paddle.arange(24, dtype=paddle.float32).reshape(
                [2, 3, 4]
            )

            pad = paddle.to_tensor([2, 1, 1], dtype="int32")
            result = F.pad(input_tensor, pad=pad, mode='circular')

            place = (
                paddle.CUDAPlace(0)
                if paddle.base.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            cir_res = exe.run(fetch_list=[result])
            np.testing.assert_allclose(cir_res[0], cir_gt)
        paddle.disable_static()

    def test_dyn_graph_reflect(self):
        x = paddle.full([10, 10], 2, dtype=paddle.float64)
        result = F.pad(x, mode='reflect', pad=(1,))
        np.testing.assert_allclose(
            result.numpy(), np.full([10, 11], 2, dtype=np.float64)
        )

    def test_special_cases(self):
        # empty padding tensor
        x = paddle.randn([10, 7], dtype=paddle.float64)
        result = F.pad(x, mode='replicate', pad=paddle.tensor([]))
        np.testing.assert_allclose(result.numpy(), x.numpy())

    def test_error_handling(self):
        dummy_x = paddle.arange(3)

        wrong_api_used = (
            "paddle.compat.nn.functional.pad() received unexpected keyword arguments 'name', 'x'. "
            "\nDid you mean to use paddle.nn.functional.pad() instead?"
        )
        ndim_no_impl = "Input tensor dimension must be in [1-5] but got {x_dim}"
        non_const_ndim_no_impl = "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now, got ndim: {x_dim}"
        mode_no_impl = "mode should be one of constant, reflect, replicate, circular, but got mirror."
        pad_len_invalid1 = "Expect len(pad) <= 6 and not -1, got: {pad_len}"
        pad_len_invalid2 = "len(pad) is bounded by input.ndim: expect len(pad) <= {max_dim}, got: {pad_len}"

        with self.assertRaises(TypeError) as cm:
            tensors = F.pad(
                x=dummy_x,
                mode='constant',
                pad=paddle.to_tensor(2),
                name='pad_layer',
            )
        self.assertEqual(str(cm.exception), wrong_api_used)

        with self.assertRaises(AssertionError) as cm:
            tensors = F.pad(
                paddle.arange(64).reshape([2] * 6),
                mode='constant',
                pad=paddle.to_tensor(2),
            )
        self.assertEqual(str(cm.exception), ndim_no_impl.format(x_dim=6))

        with self.assertRaises(ValueError) as cm:
            tensors = F.pad(paddle.arange(2), mode='circular', pad=[0, 1])
        self.assertEqual(
            str(cm.exception), non_const_ndim_no_impl.format(x_dim=1)
        )

        with self.assertRaises(AssertionError) as cm:
            tensors = F.pad(paddle.arange(2), mode='mirror', pad=[0, 1])
        self.assertEqual(str(cm.exception), mode_no_impl)

        with self.assertRaises(ValueError) as cm:
            tensors = F.pad(
                paddle.ones([2, 3, 4]),
                mode='replicate',
                pad=[0, 1, 1, 1, 1, 1, 1, 1],
            )
        self.assertEqual(str(cm.exception), pad_len_invalid1.format(pad_len=8))

        with self.assertRaises(ValueError) as cm:
            tensors = F.pad(
                paddle.ones([2, 3]), mode='replicate', pad=[0, 1, 1, 1, 1]
            )
        self.assertEqual(
            str(cm.exception), pad_len_invalid2.format(max_dim=2, pad_len=5)
        )


if __name__ == '__main__':
    unittest.main()
