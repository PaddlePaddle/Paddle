# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import warnings

import paddle


class TestTensorTypePromotion(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor([2, 3])
        self.y = paddle.to_tensor([1.0, 2.0])

    def add_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x + self.y

    def sub_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x - self.y

    def mul_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x * self.y

    def div_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x / self.y

    def test_operator(self):
        self.setUp()
        self.add_operator()
        self.sub_operator()
        self.mul_operator()
        self.div_operator()


def create_test_case(baseclass, ldtype, rdtype, expected_out_dtype=None):
    class TestPromotion(baseclass):
        def set_dtype(self):
            self.ldtype = ldtype
            self.rdtype = rdtype
            self.expected_out_dtype = expected_out_dtype

    cls_name = f"{baseclass.__name__}Between{ldtype}And{rdtype}"
    TestPromotion.__name__ = cls_name
    globals()[cls_name] = TestPromotion


class TestOperatorOverloadAddInStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.ldtype = 'float32'
        self.rdtype = 'float64'
        self.expected_out_dtype = 'float64'

    def generate_test_value(self):
        self.l_value = (paddle.randn((4, 3, 2)) * 10).astype(self.ldtype)
        self.r_value = (paddle.randn((4, 3, 2)) * 10).astype(self.rdtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value + self.r_value
            out_reverse = self.r_value + self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res

    def test_dtype_is_expected(self):
        res = self.run_api()
        self.assertEqual(res[0].dtype.__str__(), self.expected_out_dtype)
        self.assertEqual(res[1].dtype.__str__(), self.expected_out_dtype)


create_test_case(
    TestOperatorOverloadAddInStatic, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadAddInStatic, 'float32', 'float64', 'float64'
)


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadAddInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadAddInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadAddInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestOperatorOverloadSubInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value - self.r_value
            out_reverse = self.r_value - self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadSubInStatic, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadSubInStatic, 'float32', 'float64', 'float64'
)


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadSubInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadSubInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadSubInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestOperatorOverloadMulInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value * self.r_value
            out_reverse = self.r_value * self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadMulInStatic, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadMulInStatic, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadMulInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadMulInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadMulInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestOperatorOverloadGTInStatic(TestOperatorOverloadAddInStatic):
    def set_dtype(self):
        self.ldtype = 'float32'
        self.rdtype = 'float64'
        self.expected_out_dtype = 'bool'

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value > self.r_value
            out_reverse = self.r_value > self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestOperatorOverloadGTInStatic, 'float16', 'float32', 'bool')
create_test_case(TestOperatorOverloadGTInStatic, 'float16', 'float64', 'bool')

create_test_case(TestOperatorOverloadGTInStatic, 'float32', 'float64', 'bool')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadGTInStatic, 'bfloat16', 'float16', 'bool'
    )
    create_test_case(
        TestOperatorOverloadGTInStatic, 'bfloat16', 'float32', 'bool'
    )
    create_test_case(
        TestOperatorOverloadGTInStatic, 'bfloat16', 'float64', 'bool'
    )


if __name__ == '__main__':
    unittest.main()
