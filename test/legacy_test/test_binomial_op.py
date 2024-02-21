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

import math
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api


def pdf(x, n, p):
    norm = math.factorial(n) / math.factorial(n - x) / math.factorial(x)
    return norm * math.pow(p, x) * math.pow(1 - p, n - x)


def output_hist(out, n, p, a=10, b=20):
    prob = []
    bin = []
    for i in range(a, b + 1):
        prob.append(pdf(i, n, p))
        bin.append(i)
    bin.append(b + 0.1)

    hist, _ = np.histogram(out, bin)
    hist = hist.astype("float32")
    hist = hist / float(out.size)
    return hist, prob


class TestBinomialOp(OpTest):
    def setUp(self):
        self.python_api = paddle.binomial
        self.op_type = "binomial"
        self.init_dtype()
        self.config()
        self.init_test_case()
        self.inputs = {
            "count": self.count,
            "prob": self.probability,
        }
        self.attrs = {}
        self.outputs = {"out": self.out}

    def init_dtype(self):
        self.count_dtype = np.float32
        self.probability_dtype = np.float32
        self.outputs_dtype = np.int64

    def config(self):
        self.n = 20
        self.p = 0.2

    def init_test_case(self):
        self.count = np.full([2048, 1024], self.n, dtype=self.count_dtype)
        self.probability = np.full(
            [2048, 1024], self.p, dtype=self.probability_dtype
        )
        self.out = np.zeros((2048, 1024)).astype(self.outputs_dtype)

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = output_hist(np.array(outs[0]), self.n, self.p, a=5, b=15)
        # setting of `rtol` and `atol` refer to ``test_bernoulli_op``, ``test_poisson_op``
        # and ``test_multinomial_op``
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestBinomialApi(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        n = 30
        p = 0.1
        count = paddle.full([16384, 1024], n, dtype="int64")
        probability = paddle.to_tensor(p)
        out = paddle.binomial(count, probability)
        paddle.enable_static()
        hist, prob = output_hist(out.numpy(), n, p, a=5, b=25)
        # setting of `rtol` and `atol` refer to ``test_bernoulli_op``, ``test_poisson_op``
        # and ``test_multinomial_op``
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)

    @test_with_pir_api
    def test_static(self):
        n = 200
        p = 0.6
        count = paddle.to_tensor(n, dtype="int64")
        probability = paddle.full([16384, 1024], p)
        out = paddle.binomial(count, probability)
        exe = paddle.static.Executor(paddle.CPUPlace())
        out = exe.run(paddle.static.default_main_program(), fetch_list=[out])
        hist, prob = output_hist(out[0], n, p, a=70, b=140)
        # setting of `rtol` and `atol` refer to ``test_bernoulli_op``, ``test_poisson_op``
        # and ``test_multinomial_op``
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        paddle.disable_static()
        paddle.set_device('gpu')
        paddle.seed(2023)
        count = paddle.full([32, 3, 1024, 768], 100.0, dtype="float32")
        probability = paddle.to_tensor(0.4)
        y = paddle.binomial(count, probability)
        y_np = y.numpy()

        expect = [
            45,
            49,
            40,
            39,
            39,
            37,
            35,
            35,
            43,
            38,
            42,
            39,
            52,
            44,
            48,
            47,
            48,
            50,
            38,
            41,
        ]
        np.testing.assert_array_equal(y_np[0, 0, 0, 0:20], expect)

        expect = [
            43,
            35,
            35,
            35,
            43,
            35,
            45,
            38,
            39,
            45,
            39,
            46,
            52,
            41,
            54,
            41,
            40,
            49,
            38,
            40,
        ]
        np.testing.assert_array_equal(y_np[8, 1, 300, 200:220], expect)

        expect = [
            37,
            40,
            41,
            48,
            39,
            28,
            42,
            45,
            40,
            40,
            35,
            43,
            35,
            46,
            42,
            35,
            42,
            43,
            37,
            32,
        ]
        np.testing.assert_array_equal(y_np[16, 1, 600, 400:420], expect)

        expect = [
            43,
            42,
            39,
            38,
            38,
            38,
            43,
            37,
            36,
            44,
            37,
            46,
            42,
            41,
            40,
            39,
            40,
            34,
            40,
            38,
        ]
        np.testing.assert_array_equal(y_np[24, 2, 900, 600:620], expect)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestBinomialFP16Op(TestBinomialOp):
    def init_dtype(self):
        self.count_dtype = np.float16
        self.probability_dtype = np.float16
        self.outputs_dtype = np.int64

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place_customized(self.verify_output, place)

    def verify_output(self, outs):
        hist, prob = output_hist(np.array(outs[0]), self.n, self.p, a=5, b=15)
        # setting of `rtol` and `atol` refer to ``test_bernoulli_op``, ``test_poisson_op``
        # and ``test_multinomial_op``
        np.testing.assert_allclose(hist, prob, atol=0.01)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestBinomialBF16Op(TestBinomialOp):
    def init_dtype(self):
        self.probability_dtype = np.uint16
        self.count_dtype = np.uint16
        self.outputs_dtype = np.int64

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place_customized(self.verify_output, place)

    def init_test_case(self):
        self.count = convert_float_to_uint16(
            np.full([2048, 1024], self.n).astype("float32")
        )
        self.probability = convert_float_to_uint16(
            np.full([2048, 1024], self.p).astype("float32")
        )
        self.out = np.zeros((2048, 1024)).astype(self.outputs_dtype)

    def verify_output(self, outs):
        hist, prob = output_hist(np.array(outs[0]), self.n, self.p, a=5, b=15)
        # setting of `rtol` and `atol` refer to ``test_bernoulli_op``, ``test_poisson_op``
        # and ``test_multinomial_op``
        np.testing.assert_allclose(hist, prob, atol=0.01)


if __name__ == "__main__":
    unittest.main()
