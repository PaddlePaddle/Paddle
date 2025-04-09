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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_pir_only,
)

import paddle

SEED = 2020
np.random.seed(SEED)


def test_bool_cast(x):
    x = paddle.to_tensor(x)
    x = bool(x)
    return x


def test_int_cast(x):
    x = paddle.to_tensor(x)
    x = int(x)
    return x


def test_float_cast(x):
    x = paddle.to_tensor(x)
    x = float(x)
    return x


def test_not_var_cast(x):
    x = int(x)
    return x


def test_mix_cast(x):
    x = paddle.to_tensor(x)
    x = int(x)
    x = float(x)
    x = bool(x)
    x = float(x)
    return x


def test_complex_cast(x):
    x = paddle.to_tensor(x)
    x = complex(x)
    return x


def test_not_var_complex_cast(x):
    x = complex(x)
    return x


class TestCastBase(Dy2StTestBase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.prepare()

    def prepare(self):
        self.input_shape = (16, 32)
        self.input_dtype = 'float32'
        self.input = (
            np.random.binomial(4, 0.3, size=np.prod(self.input_shape))
            .reshape(self.input_shape)
            .astype(self.input_dtype)
        )
        self.cast_dtype = 'bool'

    def set_func(self):
        self.func = paddle.jit.to_static(full_graph=True)(test_bool_cast)

    def do_test(self):
        res = self.func(self.input)
        return res

    @test_ast_only  # TODO: add new sot only test.
    def test_cast_result(self):
        self.set_func()
        res = self.do_test().numpy()
        self.assertTrue(
            res.dtype == self.cast_dtype,
            msg=f'The target dtype is {self.cast_dtype}, but the casted dtype is {res.dtype}.',
        )
        ref_val = self.input.astype(self.cast_dtype)
        np.testing.assert_allclose(
            res,
            ref_val,
            rtol=1e-05,
            err_msg=f'The casted value is {res}.\nThe correct value is {ref_val}.',
        )


class TestIntCast(TestCastBase):
    def prepare(self):
        self.input_shape = (1,)
        self.input_dtype = 'float32'
        self.input = (
            np.random.normal(loc=6, scale=10, size=np.prod(self.input_shape))
            .reshape(self.input_shape)
            .astype(self.input_dtype)
        )
        self.cast_dtype = 'int32'

    def set_func(self):
        self.func = paddle.jit.to_static(full_graph=True)(test_int_cast)


class TestFloatCast(TestCastBase):
    def prepare(self):
        self.input_shape = (8, 16)
        self.input_dtype = 'bool'
        self.input = (
            np.random.binomial(2, 0.5, size=np.prod(self.input_shape))
            .reshape(self.input_shape)
            .astype(self.input_dtype)
        )
        self.cast_dtype = 'float32'

    def set_func(self):
        self.func = paddle.jit.to_static(full_graph=True)(test_float_cast)


class TestMixCast(TestCastBase):
    def prepare(self):
        self.input_shape = (8, 32)
        self.input_dtype = 'float32'
        self.input = (
            np.random.normal(loc=6, scale=10, size=np.prod(self.input_shape))
            .reshape(self.input_shape)
            .astype(self.input_dtype)
        )
        self.cast_int = 'int'
        self.cast_float = 'float32'
        self.cast_bool = 'bool'
        self.cast_dtype = 'float32'

    def set_func(self):
        self.func = paddle.jit.to_static(full_graph=True)(test_mix_cast)

    @test_ast_only  # TODO: add new symbolic only test.
    def test_cast_result(self):
        self.set_func()
        res = self.do_test().numpy()
        self.assertTrue(
            res.dtype == self.cast_dtype,
            msg=f'The target dtype is {self.cast_dtype}, but the casted dtype is {res.dtype}.',
        )
        ref_val = (
            self.input.astype(self.cast_int)
            .astype(self.cast_float)
            .astype(self.cast_bool)
            .astype(self.cast_dtype)
        )
        np.testing.assert_allclose(
            res,
            ref_val,
            rtol=1e-05,
            err_msg=f'The casted value is {res}.\nThe correct value is {ref_val}.',
        )


class TestNotVarCast(TestCastBase):
    def prepare(self):
        self.input = 3.14
        self.cast_dtype = 'int'

    def set_func(self):
        self.func = paddle.jit.to_static(full_graph=True)(test_not_var_cast)

    @test_ast_only
    def test_cast_result(self):
        self.set_func()
        res = self.do_test()
        self.assertTrue(type(res) == int, msg='The casted dtype is not int.')
        ref_val = int(self.input)
        self.assertTrue(
            res == ref_val,
            msg=f'The casted value is {res}.\nThe correct value is {ref_val}.',
        )


@unittest.skipIf(
    paddle.core.is_compiled_with_xpu(),
    "xpu does not support complex cast temporarily",
)
class TestComplexCast(TestCastBase):
    def prepare(self):
        self.input_shape = (8, 16)
        self.input_dtype = 'float32'
        self.input = (
            np.random.binomial(2, 0.5, size=np.prod(self.input_shape))
            .reshape(self.input_shape)
            .astype(self.input_dtype)
        )
        self.cast_dtype = 'complex64'

    def set_func(self):
        self.func = paddle.jit.to_static(full_graph=True)(test_complex_cast)

    @test_pir_only
    def test_cast_result(self):
        self.set_func()
        res = self.do_test().numpy()
        self.assertTrue(
            res.dtype == self.cast_dtype,
            msg=f'The target dtype is {self.cast_dtype}, but the casted dtype is {res.dtype}.',
        )
        ref_val = self.input.astype(self.cast_dtype)
        np.testing.assert_allclose(
            res,
            ref_val,
            rtol=1e-05,
            err_msg=f'The casted value is {res}.\nThe correct value is {ref_val}.',
        )


class TestNotVarComplexCast(TestCastBase):
    def prepare(self):
        self.input = 3.14
        self.cast_dtype = 'complex'

    def set_func(self):
        self.func = paddle.jit.to_static(full_graph=True)(
            test_not_var_complex_cast
        )

    @test_ast_only
    def test_cast_result(self):
        self.set_func()
        res = self.do_test()
        self.assertTrue(
            type(res) == complex, msg='The casted dtype is not complex.'
        )
        ref_val = complex(self.input)
        self.assertTrue(
            res == ref_val,
            msg=f'The casted value is {res}.\nThe correct value is {ref_val}.',
        )


if __name__ == '__main__':
    unittest.main()
