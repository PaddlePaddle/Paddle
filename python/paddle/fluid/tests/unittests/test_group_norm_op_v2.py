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

import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import _test_eager_guard
import paddle


def group_norm_naive_for_general_dimension(x, scale, bias, epsilon, groups):
    # original version group norm only support 4-D tensor
    # this function generalizes to support differnt dimensions tensor (>= 2-D)
    input_shape = x.shape
    N, C = x.shape[0], x.shape[1]
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    output = (x - mean) / np.sqrt(var + epsilon)
    output = output.reshape(input_shape) * scale.reshape(
        (-1, 1, 1)
    ) + bias.reshape((-1, 1, 1))
    return output


class TestDygraphGroupNormv2(unittest.TestCase):
    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(fluid.CUDAPlace(0))
        shapes = [
            [2, 2, 2, 2],
            [2, 2, 4],
            [4, 2],
            [4, 2, 6, 6, 2],
            [2, 2, 2, 2, 2, 2],
        ]
        for p in places:

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    gn = fluid.dygraph.GroupNorm(channels=2, groups=2)
                    y = gn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    gn = paddle.nn.GroupNorm(num_channels=2, num_groups=2)
                    y = gn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def test_weight_bias_false():
                with fluid.dygraph.guard(p):
                    gn = paddle.nn.GroupNorm(
                        num_channels=2,
                        num_groups=2,
                        weight_attr=False,
                        bias_attr=False,
                    )

            def test_nn_exception():
                with fluid.dygraph.guard(p):

                    def attr_data_format():
                        out = paddle.nn.GroupNorm(
                            num_groups=2, num_channels=2, data_format="CNHW"
                        )

                    self.assertRaises(ValueError, attr_data_format)

            for shape in shapes:
                x = np.random.randn(*shape).astype("float32")
                y1 = compute_v1(x)
                y2 = compute_v2(x)
                result = np.allclose(y1, y2, atol=1e-5)
                if not result:
                    print("y1:", y1, "\ty2:", y2)
                self.assertTrue(result)
                test_weight_bias_false()
                test_nn_exception()

    def test_static(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(fluid.CUDAPlace(0))
        shapes = [
            [2, 6, 2, 2],
            [2, 6, 4],
            [4, 6],
            [4, 6, 6, 6, 2],
            [4, 6, 2, 2, 2, 2],
        ]
        for p in places:
            exe = fluid.Executor(p)

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    gn = fluid.dygraph.GroupNorm(channels=6, groups=2)
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = gn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    gn = paddle.nn.GroupNorm(num_channels=6, num_groups=2)
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = gn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            for shape in shapes:
                x = np.random.randn(*shape).astype("float32")
                y1 = compute_v1(x)
                y2 = compute_v2(x)
                np.testing.assert_allclose(y1, y2, rtol=1e-05, atol=1e-05)

    def test_eager_api(self):
        with _test_eager_guard():
            self.test_dygraph()


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
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(fluid.CUDAPlace(0))

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

    def test_eager_api(self):
        with _test_eager_guard():
            self.test_numerical_accuracy()


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
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(fluid.CUDAPlace(0))

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

    def test_eager_api(self):
        with _test_eager_guard():
            self.test_numerical_accuracy()


class TestGroupNormDimException(unittest.TestCase):
    def test_exception(self):
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
