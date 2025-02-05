#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_uint16_to_float, paddle_static_guard

import paddle
from paddle import base
from paddle.base import core
from paddle.tensor import random


class TestGaussianRandomOp(OpTest):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.python_api = paddle.tensor.random.gaussian
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.01)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestGaussianRandomFP16Op(OpTest):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.python_api = paddle.tensor.random.gaussian
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "dtype": paddle.float16,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {'Out': np.zeros((123, 92), dtype='float16')}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float16")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float16")
        hist2 /= float(outs[0].size)
        np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.015)


def gaussian_wrapper(dtype_=np.uint16):
    def gauss_wrapper(shape, mean, std, seed, dtype=np.uint16, name=None):
        return paddle.tensor.random.gaussian(
            shape, mean, std, seed, dtype, name
        )

    return gauss_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestGaussianRandomBF16Op(OpTest):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.python_api = gaussian_wrapper(dtype_=np.uint16)
        self.__class__.op_type = self.op_type
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "dtype": paddle.bfloat16,
            "use_mkldnn": self.use_mkldnn,
        }
        paddle.seed(10)

        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.0

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        outs = convert_uint16_to_float(outs)
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.05)


class TestMeanStdAreInt(TestGaussianRandomOp):
    def set_attrs(self):
        self.mean = 1
        self.std = 2


# Situation 2: Attr(shape) is a list(with tensor)
class TestGaussianRandomOp_ShapeTensorList(TestGaussianRandomOp):
    def setUp(self):
        '''Test gaussian_random op with specified value'''
        self.op_type = "gaussian_random"
        self.python_api = paddle.tensor.random.gaussian
        self.init_data()
        shape_tensor_list = []
        for index, ele in enumerate(self.shape):
            shape_tensor_list.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )

        self.attrs = {
            'shape': self.infer_shape,
            'mean': self.mean,
            'std': self.std,
            'seed': self.seed,
            'use_mkldnn': self.use_mkldnn,
        }

        self.inputs = {"ShapeTensorList": shape_tensor_list}
        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, 92]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)


class TestGaussianRandomOp2_ShapeTensorList(
    TestGaussianRandomOp_ShapeTensorList
):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [-1, -1]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


class TestGaussianRandomOp3_ShapeTensorList(
    TestGaussianRandomOp_ShapeTensorList
):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.use_mkldnn = True
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


class TestGaussianRandomOp4_ShapeTensorList(
    TestGaussianRandomOp_ShapeTensorList
):
    def init_data(self):
        self.shape = [123, 92]
        self.infer_shape = [123, -1]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


# Situation 3: shape is a tensor
class TestGaussianRandomOp1_ShapeTensor(TestGaussianRandomOp):
    def setUp(self):
        '''Test gaussian_random op with specified value'''
        self.op_type = "gaussian_random"
        self.init_data()
        self.use_mkldnn = False
        self.python_api = paddle.tensor.random.gaussian
        self.inputs = {"ShapeTensor": np.array(self.shape).astype("int32")}
        self.attrs = {
            'mean': self.mean,
            'std': self.std,
            'seed': self.seed,
            'use_mkldnn': self.use_mkldnn,
        }
        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def init_data(self):
        self.shape = [123, 92]
        self.use_mkldnn = False
        self.mean = 1.0
        self.std = 2.0
        self.seed = 10


# Test python API
class TestGaussianRandomAPI(unittest.TestCase):
    def test_api(self):
        with paddle_static_guard():
            positive_2_int32 = paddle.tensor.fill_constant([1], "int32", 2000)

            positive_2_int64 = paddle.tensor.fill_constant([1], "int64", 500)
            shape_tensor_int32 = paddle.static.data(
                name="shape_tensor_int32", shape=[2], dtype="int32"
            )

            shape_tensor_int64 = paddle.static.data(
                name="shape_tensor_int64", shape=[2], dtype="int64"
            )

            out_1 = random.gaussian(
                shape=[2000, 500], dtype="float32", mean=0.0, std=1.0, seed=10
            )

            out_2 = random.gaussian(
                shape=[2000, positive_2_int32],
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_3 = random.gaussian(
                shape=[2000, positive_2_int64],
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_4 = random.gaussian(
                shape=shape_tensor_int32,
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_5 = random.gaussian(
                shape=shape_tensor_int64,
                dtype="float32",
                mean=0.0,
                std=1.0,
                seed=10,
            )

            out_6 = random.gaussian(
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
            out = paddle.tensor.random.gaussian([2, 3])
            self.assertEqual(out.dtype, paddle.float16)

        def test_default_fp32():
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.gaussian([2, 3])
            self.assertEqual(out.dtype, paddle.float32)

        def test_default_fp64():
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.gaussian([2, 3])
            self.assertEqual(out.dtype, paddle.float64)

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
            self.assertEqual(out.dtype, paddle.float16)

        def test_default_fp32():
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.standard_normal([2, 3])
            self.assertEqual(out.dtype, paddle.float32)

        def test_default_fp64():
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.standard_normal([2, 3])
            self.assertEqual(out.dtype, paddle.float64)

        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu')
            test_default_fp16()
        test_default_fp64()
        test_default_fp32()

    def test_complex_dtype(self):
        def test_complex64():
            out = paddle.tensor.random.standard_normal(
                [2, 3], dtype='complex64'
            )
            self.assertEqual(out.dtype, paddle.complex64)

        def test_complex128():
            out = paddle.tensor.random.standard_normal(
                [2, 3], dtype='complex128'
            )
            self.assertEqual(out.dtype, paddle.complex128)

        test_complex64()
        test_complex128()


class TestComplexRandnAPI(unittest.TestCase):
    def test_dygraph(self):
        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        with base.dygraph.guard(place):
            for dtype in ['complex64', 'complex128']:
                out = paddle.randn([5000, 2], dtype=dtype)
                mean = out.numpy().mean()
                np.testing.assert_allclose(
                    0.0 + 0.0j, mean, rtol=0.02, atol=0.02
                )
                var = out.numpy().var()
                var_real = out.numpy().real.var()
                var_imag = out.numpy().imag.var()
                np.testing.assert_allclose(var, 1.0, rtol=0.02, atol=0.02)
                np.testing.assert_allclose(var_real, 0.5, rtol=0.02, atol=0.02)
                np.testing.assert_allclose(var_imag, 0.5, rtol=0.02, atol=0.02)

    def test_static(self):
        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        with paddle_static_guard():
            for dtype in ['complex64', 'complex128']:
                main_program = paddle.static.Program()
                with paddle.static.program_guard(main_program):
                    out = paddle.randn([5000, 2], dtype=dtype)
                    exe = paddle.static.Executor(place)
                    ret = exe.run(fetch_list=[out])

                mean = ret[0].mean()
                np.testing.assert_allclose(
                    0.0 + 0.0j, mean, rtol=0.02, atol=0.02
                )
                var = ret[0].var()
                var_real = ret[0].real.var()
                var_imag = ret[0].imag.var()
                np.testing.assert_allclose(var, 1.0, rtol=0.02, atol=0.02)
                np.testing.assert_allclose(var_real, 0.5, rtol=0.02, atol=0.02)
                np.testing.assert_allclose(var_imag, 0.5, rtol=0.02, atol=0.02)


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        # Different GPU generatte different random value. Only test V100 here.
        if "V100" not in paddle.device.cuda.get_device_name():
            return

        def _check_random_value(shape, dtype, expect, expect_mean, expect_std):
            x = paddle.randn(shape, dtype=dtype)
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
            [32, 3, 1024, 1024], paddle.float64, expect, expect_mean, expect_std
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
            [32, 3, 1024, 1024], paddle.float32, expect, expect_mean, expect_std
        )

        # test randn in large shape
        expect = [
            -1.4770278,
            -0.637431,
            -0.41728288,
            0.31339037,
            -1.7627009,
            0.4061812,
            1.0679497,
            0.03405872,
            -0.7271235,
            -0.42642546,
        ]

        expect_mean = 0.0000010386128224126878194510936737060547
        expect_std = 1.00000822544097900390625
        _check_random_value(
            [4, 2, 60000, 12000],
            paddle.float32,
            expect,
            expect_mean,
            expect_std,
        )

        # test randn with seed 0 in large shape
        paddle.seed(0)
        expect = [
            -1.7653463,
            0.5957617,
            0.45865676,
            -0.3061651,
            0.17204928,
            -1.7802757,
            -0.10731091,
            1.042362,
            0.70476884,
            0.2720365,
        ]
        expect_mean = -0.0000002320642948916429304517805576324463
        expect_std = 1.00001156330108642578125
        _check_random_value(
            [4, 2, 60000, 12000],
            paddle.float32,
            expect,
            expect_mean,
            expect_std,
        )


if __name__ == "__main__":
    unittest.main()
