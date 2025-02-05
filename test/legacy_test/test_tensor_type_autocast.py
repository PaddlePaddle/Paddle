# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TestAutocastBase(unittest.TestCase):
    def setUp(self):
        self.set_api_and_dtypes()
        self.places = [paddle.CPUPlace()]
        if paddle.core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def set_api_and_dtypes(self):
        pass


def create_test_case(
    baseclass,
    api,
    support_int_types=["uint8", "int8", "int16", "int32", "int64"],
    **kwargs,
):
    class TestAutocast(baseclass):
        def set_api_and_dtypes(self):
            self.support_int_types = support_int_types
            self.api = api

        def test_dygraph(self):
            for place in self.places:
                paddle.disable_static(place)
                for type in self.support_int_types:
                    x = paddle.arange(-100, 100).astype(type)
                    x_float = x.astype("float32")
                    int_out = self.api(x, **kwargs)
                    float_out = self.api(x_float, **kwargs)
                    np.testing.assert_array_equal(
                        int_out.numpy(), float_out.numpy()
                    )

        def test_static(self):
            paddle.enable_static()
            for place in self.places:
                exe = paddle.static.Executor(place)
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                for type in self.support_int_types:
                    with paddle.static.program_guard(
                        main_program, startup_program
                    ):
                        x = paddle.arange(-100, 100).astype(type)
                        x_float = x.astype("float32")
                        int_out = self.api(x, **kwargs)
                        float_out = self.api(x_float, **kwargs)
                        out = exe.run(fetch_list=[int_out, float_out])
                    np.testing.assert_array_equal(out[0], out[1])
            paddle.disable_static(place)

    api_name = api.__name__
    cls_name = f"{baseclass.__name__}{api_name}"
    TestAutocast.__name__ = cls_name
    globals()[cls_name] = TestAutocast


def create_test_case_with_grad(
    baseclass,
    api,
    support_int_types=["uint8", "int8", "int16", "int32", "int64"],
    **kwargs,
):
    class TestAutocastValidGrad(baseclass):
        def set_api_and_dtypes(self):
            self.support_int_types = support_int_types
            self.api = api

        def test_dygraph(self):
            for place in self.places:
                paddle.disable_static(place)
                for type in self.support_int_types:
                    x = paddle.arange(-100, 100).astype(type)
                    x_float = x.astype("float32")
                    x.stop_gradient = False
                    x_float.stop_gradient = False
                    int_out = self.api(x, **kwargs)
                    float_out = self.api(x_float, **kwargs)
                    int_out.backward()
                    float_out.backward()
                    np.testing.assert_array_equal(
                        int_out.numpy(), float_out.numpy()
                    )
                    np.testing.assert_equal(x.grad.dtype, x.dtype)
                    np.testing.assert_allclose(
                        x.grad.numpy(), x_float.grad.numpy()
                    )

        def test_static(self):
            paddle.enable_static()
            for place in self.places:
                exe = paddle.static.Executor(place)
                main_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                for type in self.support_int_types:
                    with paddle.static.program_guard(
                        main_program, startup_program
                    ):
                        x = paddle.arange(-100, 100).astype(type)
                        x_float = x.astype("float32")
                        x.stop_gradient = False
                        x_float.stop_gradient = False
                        int_out = self.api(x, **kwargs)
                        float_out = self.api(x_float, **kwargs)
                        x_grad = paddle.static.gradients(int_out, x)
                        x_float_grad = paddle.static.gradients(
                            float_out, x_float
                        )
                        out = exe.run(
                            fetch_list=[
                                int_out,
                                float_out,
                                x_grad,
                                x_float_grad,
                            ]
                        )
                    np.testing.assert_array_equal(out[0], out[1])
                    np.testing.assert_equal(out[2].dtype, np.dtype(type))
                    np.testing.assert_allclose(out[2], out[3])
            paddle.disable_static(place)

    api_name = api.__name__
    cls_name = f"{baseclass.__name__}{api_name}"
    TestAutocastValidGrad.__name__ = cls_name
    globals()[cls_name] = TestAutocastValidGrad


create_test_case(TestAutocastBase, paddle.acos)

create_test_case(TestAutocastBase, paddle.acosh)

create_test_case(TestAutocastBase, paddle.asin)

create_test_case(TestAutocastBase, paddle.asinh)

create_test_case(TestAutocastBase, paddle.atan)

create_test_case(TestAutocastBase, paddle.atanh)

create_test_case(TestAutocastBase, paddle.cos)

create_test_case(TestAutocastBase, paddle.cosh)

create_test_case(TestAutocastBase, paddle.digamma)

create_test_case(TestAutocastBase, paddle.erf)

create_test_case(TestAutocastBase, paddle.erfinv)

create_test_case(TestAutocastBase, paddle.i0)

create_test_case(TestAutocastBase, paddle.i0e)

create_test_case(TestAutocastBase, paddle.i1)

create_test_case(TestAutocastBase, paddle.i1e)

create_test_case(TestAutocastBase, paddle.lgamma)

create_test_case(TestAutocastBase, paddle.logcumsumexp)

create_test_case(TestAutocastBase, paddle.logit)

create_test_case(TestAutocastBase, paddle.logsumexp)

create_test_case(TestAutocastBase, paddle.polygamma, n=1)

create_test_case(TestAutocastBase, paddle.reciprocal)

create_test_case(TestAutocastBase, paddle.rsqrt)

create_test_case(TestAutocastBase, paddle.sin)

create_test_case(TestAutocastBase, paddle.sinh)

create_test_case(TestAutocastBase, paddle.nn.functional.sigmoid)

create_test_case(TestAutocastBase, paddle.sqrt)

create_test_case(TestAutocastBase, paddle.stanh)

create_test_case(TestAutocastBase, paddle.tan)

create_test_case(TestAutocastBase, paddle.tanh)

create_test_case_with_grad(TestAutocastBase, paddle.ceil)

create_test_case_with_grad(TestAutocastBase, paddle.floor)


if __name__ == '__main__':
    unittest.main()
