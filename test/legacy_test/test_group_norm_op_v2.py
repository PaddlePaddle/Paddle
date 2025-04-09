# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core


def group_norm_naive_for_general_dimension(
    x, scale, bias, epsilon, groups, channel_last=False
):
    # original version group norm only support 4-D tensor
    # this function generalizes to support different dimensions tensor (>= 2-D)
    if channel_last:
        shape = list(range(x.ndim))
        shape.insert(1, shape.pop(-1))
        x = x.transpose(shape)
    input_shape = x.shape
    N, C = x.shape[0], x.shape[1]
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    output = (x - mean) / np.sqrt(var + epsilon)
    output = output.reshape(input_shape) * scale.reshape(
        [-1] + [1] * (x.ndim - 2)
    ) + bias.reshape([-1] + [1] * (x.ndim - 2))
    if channel_last:
        shape = list(range(output.ndim))
        shape.insert(len(shape), shape.pop(1))
        output = output.transpose(shape)
    return output


class TestGroupNormAPIV2_With_General_Dimensions(unittest.TestCase):
    def test_numerical_accuracy(self):
        paddle.disable_static()
        shapes = [
            (2, 6),
            (2, 6, 4),
            (2, 6, 4, 4),
            (2, 6, 6, 6, 2),
            (2, 6, 6, 6, 2, 3),
        ]
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            for shape in shapes:
                scale = np.array([1]).astype("float32")
                bias = np.array([0]).astype("float32")
                data = np.random.random(shape).astype("float32")
                expect_res1 = group_norm_naive_for_general_dimension(
                    data, scale, bias, epsilon=1e-5, groups=6
                )
                expect_res2 = group_norm_naive_for_general_dimension(
                    data, scale, bias, epsilon=1e-5, groups=2
                )

                gn1 = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
                gn2 = paddle.nn.GroupNorm(num_channels=6, num_groups=2)
                data_pd = paddle.to_tensor(data)
                result1 = gn1(data_pd).numpy()
                result2 = gn2(data_pd).numpy()
                self.assertTrue(np.allclose(result1, expect_res1, atol=1e-5))
                self.assertTrue(np.allclose(result2, expect_res2, atol=1e-5))


class TestGroupNormAPIV2_With_NCL(unittest.TestCase):
    def test_numerical_accuracy(self):
        paddle.disable_static()
        shape = (2, 6, 4)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NCL'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NCL'
            )
            data_pd = paddle.to_tensor(data)
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(result1, expect_res1, atol=1e-5)
            np.testing.assert_allclose(result2, expect_res2, atol=1e-5)


class TestGroupNormAPIV2_With_NCDHW(unittest.TestCase):
    def test_numerical_accuracy(self):
        paddle.disable_static()
        shape = (2, 6, 4, 2, 2)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NCDHW'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NCDHW'
            )
            data_pd = paddle.to_tensor(data)
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(result1, expect_res1, atol=1e-5)
            np.testing.assert_allclose(result2, expect_res2, atol=1e-5)


class TestGroupNormAPIV2_With_NLC(unittest.TestCase):
    def test_numerical_accuracy(self):
        paddle.disable_static()
        shape = (2, 4, 6)
        np.random.seed(10)
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6, channel_last=True
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2, channel_last=True
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NLC'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NLC'
            )
            data_pd = paddle.to_tensor(data)
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(result1, expect_res1, atol=1e-5)
            np.testing.assert_allclose(result2, expect_res2, atol=1e-5)


class TestGroupNormAPIV2_With_NHWC(unittest.TestCase):
    def test_numerical_accuracy(self):
        paddle.disable_static()
        shape = (2, 4, 2, 6)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6, channel_last=True
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2, channel_last=True
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NHWC'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NHWC'
            )
            data_pd = paddle.to_tensor(data)
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(result1, expect_res1, atol=1e-5)
            np.testing.assert_allclose(result2, expect_res2, atol=1e-5)


class TestGroupNormAPIV2_With_NDHWC(unittest.TestCase):
    def test_numerical_accuracy(self):
        paddle.disable_static()
        shape = (2, 4, 2, 2, 6)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6, channel_last=True
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2, channel_last=True
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NDHWC'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NDHWC'
            )
            data_pd = paddle.to_tensor(data)
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(result1, expect_res1, atol=1e-5)
            np.testing.assert_allclose(result2, expect_res2, atol=1e-5)


class TestGroupNormAPIV2_With_General_Dimensions_fp16(unittest.TestCase):
    def test_numerical_accuracy(self):
        # fp16 only supported in cuda
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        shapes = [
            (2, 6, 4),
            (2, 6, 4, 4),
            (2, 6, 6, 6, 2),
            (2, 6, 6, 6, 2, 3),
            (2, 6, 6, 6, 256, 3),
        ]
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            for shape in shapes:
                scale = np.array([1]).astype("float32")
                bias = np.array([0]).astype("float32")
                data = np.random.random(shape).astype("float32")
                expect_res1 = group_norm_naive_for_general_dimension(
                    data, scale, bias, epsilon=1e-5, groups=6
                )
                expect_res2 = group_norm_naive_for_general_dimension(
                    data, scale, bias, epsilon=1e-5, groups=2
                )

                gn1 = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
                gn2 = paddle.nn.GroupNorm(num_channels=6, num_groups=2)
                paddle.assign(paddle.cast(gn1.weight, 'float16'), gn1.weight)
                paddle.assign(paddle.cast(gn1.bias, 'float16'), gn1.bias)
                paddle.assign(paddle.cast(gn2.weight, 'float16'), gn2.weight)
                paddle.assign(paddle.cast(gn2.bias, 'float16'), gn2.bias)

                data_pd = paddle.to_tensor(data.astype('float16'))
                result1 = gn1(data_pd).numpy()
                result2 = gn2(data_pd).numpy()
                np.testing.assert_allclose(
                    result1, expect_res1, rtol=1e-2, atol=1e-3
                )
                np.testing.assert_allclose(
                    result2, expect_res2, rtol=1e-2, atol=1e-3
                )


class TestGroupNormAPIV2_With_NCL_fp16(unittest.TestCase):
    def test_numerical_accuracy(self):
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        shape = (2, 6, 4)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NCL'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NCL'
            )
            paddle.assign(paddle.cast(gn1.weight, 'float16'), gn1.weight)
            paddle.assign(paddle.cast(gn1.bias, 'float16'), gn1.bias)
            paddle.assign(paddle.cast(gn2.weight, 'float16'), gn2.weight)
            paddle.assign(paddle.cast(gn2.bias, 'float16'), gn2.bias)

            data_pd = paddle.to_tensor(data.astype('float16'))
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(
                result1, expect_res1, rtol=1e-2, atol=1e-3
            )
            np.testing.assert_allclose(
                result2, expect_res2, rtol=1e-2, atol=1e-3
            )


class TestGroupNormAPIV2_With_NCDHW_fp16(unittest.TestCase):
    def test_numerical_accuracy(self):
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        shape = (2, 6, 4, 2, 2)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NCDHW'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NCDHW'
            )
            paddle.assign(paddle.cast(gn1.weight, 'float16'), gn1.weight)
            paddle.assign(paddle.cast(gn1.bias, 'float16'), gn1.bias)
            paddle.assign(paddle.cast(gn2.weight, 'float16'), gn2.weight)
            paddle.assign(paddle.cast(gn2.bias, 'float16'), gn2.bias)

            data_pd = paddle.to_tensor(data.astype('float16'))
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(
                result1, expect_res1, rtol=1e-2, atol=1e-2
            )
            np.testing.assert_allclose(
                result2, expect_res2, rtol=1e-2, atol=1e-2
            )


class TestGroupNormAPIV2_With_NLC_fp16(unittest.TestCase):
    def test_numerical_accuracy(self):
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        shape = (2, 4, 6)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6, channel_last=True
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2, channel_last=True
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NLC'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NLC'
            )
            paddle.assign(paddle.cast(gn1.weight, 'float16'), gn1.weight)
            paddle.assign(paddle.cast(gn1.bias, 'float16'), gn1.bias)
            paddle.assign(paddle.cast(gn2.weight, 'float16'), gn2.weight)
            paddle.assign(paddle.cast(gn2.bias, 'float16'), gn2.bias)

            data_pd = paddle.to_tensor(data.astype('float16'))
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(
                result1, expect_res1, rtol=1e-2, atol=1e-3
            )
            np.testing.assert_allclose(
                result2, expect_res2, rtol=1e-2, atol=1e-3
            )


class TestGroupNormAPIV2_With_NHWC_fp16(unittest.TestCase):
    def test_numerical_accuracy(self):
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        shape = (2, 4, 2, 6)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6, channel_last=True
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2, channel_last=True
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NHWC'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NHWC'
            )
            paddle.assign(paddle.cast(gn1.weight, 'float16'), gn1.weight)
            paddle.assign(paddle.cast(gn1.bias, 'float16'), gn1.bias)
            paddle.assign(paddle.cast(gn2.weight, 'float16'), gn2.weight)
            paddle.assign(paddle.cast(gn2.bias, 'float16'), gn2.bias)

            data_pd = paddle.to_tensor(data.astype('float16'))
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(
                result1, expect_res1, rtol=1e-2, atol=1e-3
            )
            np.testing.assert_allclose(
                result2, expect_res2, rtol=1e-2, atol=1e-3
            )


class TestGroupNormAPIV2_With_NDHWC_fp16(unittest.TestCase):
    def test_numerical_accuracy(self):
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        shape = (2, 4, 2, 2, 6)
        np.random.seed(10)
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(base.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            scale = np.array([1]).astype("float32")
            bias = np.array([0]).astype("float32")
            data = np.random.random(shape).astype("float32")
            expect_res1 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=6, channel_last=True
            )
            expect_res2 = group_norm_naive_for_general_dimension(
                data, scale, bias, epsilon=1e-5, groups=2, channel_last=True
            )

            gn1 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=6, data_format='NDHWC'
            )
            gn2 = paddle.nn.GroupNorm(
                num_channels=6, num_groups=2, data_format='NDHWC'
            )
            paddle.assign(paddle.cast(gn1.weight, 'float16'), gn1.weight)
            paddle.assign(paddle.cast(gn1.bias, 'float16'), gn1.bias)
            paddle.assign(paddle.cast(gn2.weight, 'float16'), gn2.weight)
            paddle.assign(paddle.cast(gn2.bias, 'float16'), gn2.bias)

            data_pd = paddle.to_tensor(data.astype('float16'))
            result1 = gn1(data_pd).numpy()
            result2 = gn2(data_pd).numpy()
            np.testing.assert_allclose(
                result1, expect_res1, rtol=1e-2, atol=1e-2
            )
            np.testing.assert_allclose(
                result2, expect_res2, rtol=1e-2, atol=1e-2
            )


class TestGroupNormDimException(unittest.TestCase):
    def test_exception(self):
        with paddle.pir_utils.OldIrGuard():

            def test_empty_input_static_API():
                x = paddle.to_tensor([], dtype='float32')
                paddle.static.nn.group_norm(x, 3)

            self.assertRaises(ValueError, test_empty_input_static_API)

            def test_one_dim_input_static_API():
                x = paddle.randn((3,), dtype='float32')
                paddle.static.nn.group_norm(x, 3)

            self.assertRaises(ValueError, test_one_dim_input_static_API)


if __name__ == '__main__':
    unittest.main()
