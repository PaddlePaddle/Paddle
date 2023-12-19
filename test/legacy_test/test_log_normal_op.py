#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import paddle_static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.tensor import random

SEED = 100
np.random.seed(SEED)
paddle.seed(SEED)


def output_log_normal(shape, mean, std):
    return np.exp(np.random.normal(mean, std, shape))


class TestLogNormalAPI(unittest.TestCase):
    def test_static_api(self):
        with paddle_static_guard():
            positive_2_int32 = paddle.tensor.fill_constant([1], "int32", 2000)

            positive_2_int64 = paddle.tensor.fill_constant([1], "int64", 500)
            shape_tensor_int32 = paddle.static.data(
                name="shape_tensor_int32", shape=[2], dtype="int32"
            )

            shape_tensor_int64 = paddle.static.data(
                name="shape_tensor_int64", shape=[2], dtype="int64"
            )

            out_1 = random.log_normal(
                shape=[2000, 500], dtype="float32", mean=0.0, std=1.0, seed=10
            )

            out_2 = random.log_normal(
                shape=[2000, positive_2_int32],
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_3 = random.log_normal(
                shape=[2000, positive_2_int64],
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_4 = random.log_normal(
                shape=shape_tensor_int32,
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_5 = random.log_normal(
                shape=shape_tensor_int64,
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_6 = random.log_normal(
                shape=shape_tensor_int64,
                dtype=np.float32,
                mean=0.0,
                std=1.0,
                seed=10,
            )

            exe = base.Executor(place=base.CPUPlace())
            res_1, res_2, res_3, res_4, res_5, res_6 = exe.run(
                base.default_main_program(),
                feed={
                    "shape_tensor_int32": np.array([2000, 500]).astype("int32"),
                    "shape_tensor_int64": np.array([2000, 500]).astype("int64"),
                },
                fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6],
            )

            self.assertAlmostEqual(np.mean(res_1), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(res_1), 1.0, delta=0.1)
            self.assertAlmostEqual(np.mean(res_2), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(res_2), 1.0, delta=0.1)
            self.assertAlmostEqual(np.mean(res_3), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(res_3), 1.0, delta=0.1)
            self.assertAlmostEqual(np.mean(res_4), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(res_5), 1.0, delta=0.1)
            self.assertAlmostEqual(np.mean(res_5), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(res_5), 1.0, delta=0.1)
            self.assertAlmostEqual(np.mean(res_6), 0.0, delta=0.1)
            self.assertAlmostEqual(np.std(res_6), 1.0, delta=0.1)

    def test_default_dtype(self):
        def test_default_fp16():
            paddle.framework.set_default_dtype('float16')
            out = paddle.tensor.random.log_normal([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP16)

        def test_default_fp32():
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.log_normal([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP32)

        def test_default_fp64():
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.log_normal([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP64)

        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            test_default_fp16()
        test_default_fp64()
        test_default_fp32()


class TestStandardNormalDtype(unittest.TestCase):
    def test_default_dtype(self):
        def test_default_fp16():
            paddle.framework.set_default_dtype('float16')
            out = paddle.tensor.random.standard_normal([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP16)

        def test_default_fp32():
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.standard_normal([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP32)

        def test_default_fp64():
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.standard_normal([2, 3])
            self.assertEqual(out.dtype, base.core.VarDesc.VarType.FP64)

        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            test_default_fp16()
        test_default_fp64()
        test_default_fp32()


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        # Different GPU generatte different random value. Only test V100 here.
        if "V100" not in paddle.device.cuda.get_device_name():
            return

        def _check_random_value(dtype, expect, expect_mean, expect_std):
            x = paddle.randn([32, 3, 1024, 1024], dtype=dtype)
            actual = x.numpy()
            np.testing.assert_allclose(
                actual[2, 1, 512, 1000:1010], expect, rtol=1e-05
            )
            self.assertTrue(np.mean(actual), expect_mean)
            self.assertTrue(np.std(actual), expect_std)

        print("Test Fixed Random number on V100 GPU------>")
        paddle.disable_static()
        paddle.set_device('gpu')
        paddle.seed(2021)
        expect = [
            -0.79037829,
            -0.54411126,
            -0.32266671,
            0.35791815,
            1.44169267,
            -0.87785644,
            -1.23909874,
            -2.18194139,
            0.49489656,
            0.40703062,
        ]
        expect_mean = (
            -0.0000053026194133403266873214888799115129813799285329878330230713
        )
        expect_std = 0.99999191058126390974081232343451119959354400634765625
        _check_random_value(
            core.VarDesc.VarType.FP64, expect, expect_mean, expect_std
        )

        expect = [
            -0.7988942,
            1.8644791,
            0.02782744,
            1.3692524,
            0.6419724,
            0.12436751,
            0.12058455,
            -1.9984808,
            1.5635862,
            0.18506318,
        ]
        expect_mean = -0.00004762359094456769526004791259765625
        expect_std = 0.999975681304931640625
        _check_random_value(
            core.VarDesc.VarType.FP32, expect, expect_mean, expect_std
        )


class TestLogNormalErrors(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            mean = [1, 2, 3]
            self.assertRaises(TypeError, paddle.log_normal, mean)

            std = [1, 2, 3]
            self.assertRaises(TypeError, paddle.log_normal, std=std)

            mean = paddle.static.data('Mean', [100], 'int32')
            self.assertRaises(TypeError, paddle.log_normal, mean)

            std = paddle.static.data('Std', [100], 'int32')
            self.assertRaises(TypeError, paddle.log_normal, mean=1.0, std=std)

            self.assertRaises(TypeError, paddle.log_normal, shape=1)

            self.assertRaises(TypeError, paddle.log_normal, shape=[1.0])

            shape = paddle.static.data('Shape', [100], 'float32')
            self.assertRaises(TypeError, paddle.log_normal, shape=shape)


if __name__ == "__main__":
    unittest.main()
