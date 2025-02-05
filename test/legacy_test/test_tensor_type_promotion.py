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
        self.x = paddle.to_tensor([2, 3], dtype='float16')
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


class TestOperatorOverloadAddInDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.set_dtype()

    def set_dtype(self):
        self.ldtype = 'float32'
        self.rdtype = 'float64'
        self.expected_out_dtype = 'float64'

    def generate_test_value(self):
        self.l_value = (paddle.randn((4, 3, 2)) * 10).astype(self.ldtype)
        self.r_value = (paddle.randn((4, 3, 2)) * 10).astype(self.rdtype)

    def run_api(self):
        self.generate_test_value()

        out = self.l_value + self.r_value
        out_reverse = self.r_value + self.l_value

        return out, out_reverse

    def test_dtype_is_expected(self):
        out, out_reverse = self.run_api()
        self.assertEqual(
            out.dtype.__str__(), "paddle." + self.expected_out_dtype
        )
        self.assertEqual(
            out_reverse.dtype.__str__(), "paddle." + self.expected_out_dtype
        )


create_test_case(
    TestOperatorOverloadAddInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadAddInDygraph, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadAddInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadAddInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadAddInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestOperatorOverloadAddInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadAddInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInDygraph, 'complex128', 'float64', 'complex128'
)


class TestAPIAddInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.add(self.l_value, self.r_value)
        out_reverse = paddle.add(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIAddInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIAddInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIAddInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIAddInDygraph, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIAddInDygraph, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIAddInDygraph, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPIAddInDygraph, 'bfloat16', 'complex64', 'complex64')
    create_test_case(
        TestAPIAddInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPIAddInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'float16', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'float32', 'complex64')
create_test_case(TestAPIAddInDygraph, 'complex64', 'float64', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPIAddInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'uint8', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'int16', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'int32', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'int64', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'float16', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'float32', 'complex128')
create_test_case(TestAPIAddInDygraph, 'complex128', 'float64', 'complex128')


class TestAPIAddInplaceInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.add_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.add_(self.l_value)

        return out, out_reverse


create_test_case(TestAPIAddInplaceInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIAddInplaceInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIAddInplaceInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPIAddInplaceInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPIAddInplaceInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPIAddInplaceInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestAPIAddInplaceInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestAPIAddInplaceInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPIAddInplaceInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIAddInplaceInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPIAddInplaceInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPIAddInplaceInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPIAddInplaceInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIAddInplaceInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(TestAPIAddInplaceInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIAddInplaceInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestAPIAddInplaceInDygraph, 'complex128', 'float64', 'complex128'
)


class TestOperatorOverloadSubInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value - self.r_value
        out_reverse = self.r_value - self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadSubInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadSubInDygraph, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadSubInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadSubInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadSubInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestOperatorOverloadSubInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadSubInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInDygraph, 'complex128', 'float64', 'complex128'
)


class TestAPISubInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.subtract(self.l_value, self.r_value)
        out_reverse = paddle.subtract(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPISubInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPISubInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPISubInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPISubInDygraph, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPISubInDygraph, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPISubInDygraph, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPISubInDygraph, 'bfloat16', 'complex64', 'complex64')
    create_test_case(
        TestAPISubInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPISubInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'float16', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'float32', 'complex64')
create_test_case(TestAPISubInDygraph, 'complex64', 'float64', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPISubInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'uint8', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'int16', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'int32', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'int64', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'float16', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'float32', 'complex128')
create_test_case(TestAPISubInDygraph, 'complex128', 'float64', 'complex128')


class TestAPISubInplaceInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.subtract_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.subtract_(self.l_value)

        return out, out_reverse


create_test_case(TestAPISubInplaceInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPISubInplaceInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPISubInplaceInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPISubInplaceInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPISubInplaceInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPISubInplaceInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestAPISubInplaceInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestAPISubInplaceInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPISubInplaceInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPISubInplaceInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPISubInplaceInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPISubInplaceInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPISubInplaceInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPISubInplaceInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(
    TestAPISubInplaceInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(TestAPISubInplaceInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPISubInplaceInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(
    TestAPISubInplaceInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestAPISubInplaceInDygraph, 'complex128', 'float64', 'complex128'
)


class TestOperatorOverloadMulInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value * self.r_value
        out_reverse = self.r_value * self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadMulInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadMulInDygraph, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadMulInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadMulInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadMulInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestOperatorOverloadMulInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadMulInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInDygraph, 'complex128', 'float64', 'complex128'
)


class TestAPIMulInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.multiply(self.l_value, self.r_value)
        out_reverse = paddle.multiply(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIMulInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIMulInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIMulInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIMulInDygraph, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIMulInDygraph, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIMulInDygraph, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPIMulInDygraph, 'bfloat16', 'complex64', 'complex64')
    create_test_case(
        TestAPIMulInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPIMulInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'float16', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'float32', 'complex64')
create_test_case(TestAPIMulInDygraph, 'complex64', 'float64', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPIMulInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'uint8', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'int16', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'int32', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'int64', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'float16', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'float32', 'complex128')
create_test_case(TestAPIMulInDygraph, 'complex128', 'float64', 'complex128')


class TestAPIMulInplaceInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.multiply_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.multiply_(self.l_value)

        return out, out_reverse


create_test_case(TestAPIMulInplaceInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIMulInplaceInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIMulInplaceInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPIMulInplaceInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPIMulInplaceInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPIMulInplaceInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestAPIMulInplaceInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestAPIMulInplaceInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPIMulInplaceInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIMulInplaceInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPIMulInplaceInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPIMulInplaceInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPIMulInplaceInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIMulInplaceInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(TestAPIMulInplaceInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIMulInplaceInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestAPIMulInplaceInDygraph, 'complex128', 'float64', 'complex128'
)


class TestOperatorOverloadDivInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value / self.r_value
        out_reverse = self.r_value / self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadDivInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadDivInDygraph, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadDivInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadDivInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadDivInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestOperatorOverloadDivInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadDivInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInDygraph, 'complex128', 'float64', 'complex128'
)


class TestAPIDivInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.divide(self.l_value, self.r_value)
        out_reverse = paddle.divide(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIDivInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIDivInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIDivInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIDivInDygraph, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIDivInDygraph, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIDivInDygraph, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPIDivInDygraph, 'bfloat16', 'complex64', 'complex64')
    create_test_case(
        TestAPIDivInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPIDivInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'float16', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'float32', 'complex64')
create_test_case(TestAPIDivInDygraph, 'complex64', 'float64', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPIDivInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'uint8', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'int16', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'int32', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'int64', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'float16', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'float32', 'complex128')
create_test_case(TestAPIDivInDygraph, 'complex128', 'float64', 'complex128')


class TestAPIDivInplaceInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.divide_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.divide_(self.l_value)

        return out, out_reverse


create_test_case(TestAPIDivInplaceInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIDivInplaceInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIDivInplaceInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPIDivInplaceInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPIDivInplaceInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPIDivInplaceInDygraph, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestAPIDivInplaceInDygraph, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestAPIDivInplaceInDygraph, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(TestAPIDivInplaceInDygraph, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIDivInplaceInDygraph, 'complex64', 'int8', 'complex64')
create_test_case(TestAPIDivInplaceInDygraph, 'complex64', 'uint8', 'complex64')
create_test_case(TestAPIDivInplaceInDygraph, 'complex64', 'int16', 'complex64')
create_test_case(TestAPIDivInplaceInDygraph, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIDivInplaceInDygraph, 'complex64', 'int64', 'complex64')
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex64', 'complex128', 'complex128'
)

create_test_case(TestAPIDivInplaceInDygraph, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIDivInplaceInDygraph, 'complex128', 'int8', 'complex128')
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestAPIDivInplaceInDygraph, 'complex128', 'float64', 'complex128'
)


class TestOperatorOverloadPowInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value**self.r_value
        out_reverse = self.r_value**self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadPowInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadPowInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadPowInDygraph, 'float32', 'float64', 'float64'
)


class TestAPIPowInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.pow(self.l_value, self.r_value)
        out_reverse = paddle.pow(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIPowInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIPowInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIPowInDygraph, 'float32', 'float64', 'float64')


class TestOperatorOverloadFloorDivInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value // self.r_value
        out_reverse = self.r_value // self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadFloorDivInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadFloorDivInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadFloorDivInDygraph, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadFloorDivInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadFloorDivInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadFloorDivInDygraph, 'bfloat16', 'float64', 'float64'
    )


class TestAPIFloorDivInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.floor_divide(self.l_value, self.r_value)
        out_reverse = paddle.floor_divide(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIFloorDivInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIFloorDivInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIFloorDivInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIFloorDivInDygraph, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIFloorDivInDygraph, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIFloorDivInDygraph, 'bfloat16', 'float64', 'float64')


class TestAPIFloorDivInplaceInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.floor_divide_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.floor_divide_(self.l_value)

        return out, out_reverse


create_test_case(
    TestAPIFloorDivInplaceInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestAPIFloorDivInplaceInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestAPIFloorDivInplaceInDygraph, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPIFloorDivInplaceInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPIFloorDivInplaceInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPIFloorDivInplaceInDygraph, 'bfloat16', 'float64', 'float64'
    )


class TestOperatorOverloadModInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value % self.r_value
        out_reverse = self.r_value % self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadModInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadModInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadModInDygraph, 'float32', 'float64', 'float64'
)


class TestAPIModInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.mod(self.l_value, self.r_value)
        out_reverse = paddle.mod(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIModInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIModInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIModInDygraph, 'float32', 'float64', 'float64')


class TestAPIModInplaceInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.mod_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.mod_(self.l_value)

        return out, out_reverse


create_test_case(TestAPIModInplaceInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIModInplaceInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIModInplaceInDygraph, 'float32', 'float64', 'float64')


class TestOperatorOverloadEqualInDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.set_dtype()

    def set_dtype(self):
        self.ldtype = 'float32'
        self.rdtype = 'float64'
        self.expected_out_dtype = 'bool'

    def generate_test_value(self):
        self.l_value = (paddle.randn((4, 3, 2)) * 10).astype(self.ldtype)
        self.r_value = (paddle.randn((4, 3, 2)) * 10).astype(self.rdtype)

    def run_api(self):
        self.generate_test_value()

        out = self.l_value == self.r_value
        out_reverse = self.r_value == self.l_value

        return out, out_reverse

    def test_dtype_is_expected(self):
        out, out_reverse = self.run_api()
        self.assertEqual(
            out.dtype.__str__(), "paddle." + self.expected_out_dtype
        )
        self.assertEqual(
            out_reverse.dtype.__str__(), "paddle." + self.expected_out_dtype
        )


create_test_case(
    TestOperatorOverloadEqualInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadEqualInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadEqualInDygraph, 'float32', 'float64', 'bool'
)


class TestAPIEqualInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.equal(self.l_value, self.r_value)
        out_reverse = paddle.equal(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIEqualInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPIEqualInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPIEqualInDygraph, 'float32', 'float64', 'bool')


class TestAPIEqualInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.equal_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.equal_(self.l_value)

        return out, out_reverse


create_test_case(TestAPIEqualInplaceInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPIEqualInplaceInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPIEqualInplaceInDygraph, 'float32', 'float64', 'bool')


class TestOperatorOverloadNotEqualInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value != self.r_value
        out_reverse = self.r_value != self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadNotEqualInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadNotEqualInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadNotEqualInDygraph, 'float32', 'float64', 'bool'
)


class TestAPINotEqualInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.not_equal(self.l_value, self.r_value)
        out_reverse = paddle.not_equal(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPINotEqualInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPINotEqualInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPINotEqualInDygraph, 'float32', 'float64', 'bool')


class TestAPINotEqualInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.not_equal_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.not_equal_(self.l_value)

        return out, out_reverse


create_test_case(TestAPINotEqualInplaceInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPINotEqualInplaceInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPINotEqualInplaceInDygraph, 'float32', 'float64', 'bool')


class TestOperatorOverloadLessThanInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value < self.r_value
        out_reverse = self.r_value < self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadLessThanInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadLessThanInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadLessThanInDygraph, 'float32', 'float64', 'bool'
)


class TestAPILessThanInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.less_than(self.l_value, self.r_value)
        out_reverse = paddle.less_than(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPILessThanInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILessThanInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILessThanInDygraph, 'float32', 'float64', 'bool')


class TestAPILessThanInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.less_than_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.less_than_(self.l_value)

        return out, out_reverse


create_test_case(TestAPILessThanInplaceInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILessThanInplaceInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILessThanInplaceInDygraph, 'float32', 'float64', 'bool')


class TestOperatorOverloadLessEqualInDygraph(
    TestOperatorOverloadEqualInDygraph
):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value <= self.r_value
        out_reverse = self.r_value <= self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadLessEqualInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadLessEqualInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadLessEqualInDygraph, 'float32', 'float64', 'bool'
)


class TestAPILessEqualInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.less_equal(self.l_value, self.r_value)
        out_reverse = paddle.less_equal(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPILessEqualInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILessEqualInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILessEqualInDygraph, 'float32', 'float64', 'bool')


class TestAPILessEqualInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.less_equal_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.less_equal_(self.l_value)

        return out, out_reverse


create_test_case(TestAPILessEqualInplaceInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILessEqualInplaceInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILessEqualInplaceInDygraph, 'float32', 'float64', 'bool')


class TestOperatorOverloadGreaterThanInDygraph(
    TestOperatorOverloadEqualInDygraph
):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value > self.r_value
        out_reverse = self.r_value > self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadGreaterThanInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadGreaterThanInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadGreaterThanInDygraph, 'float32', 'float64', 'bool'
)


class TestAPIGreaterThanInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.greater_than(self.l_value, self.r_value)
        out_reverse = paddle.greater_than(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIGreaterThanInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPIGreaterThanInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPIGreaterThanInDygraph, 'float32', 'float64', 'bool')


class TestAPIGreaterThanInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.greater_than_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.greater_than_(self.l_value)

        return out, out_reverse


create_test_case(
    TestAPIGreaterThanInplaceInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestAPIGreaterThanInplaceInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestAPIGreaterThanInplaceInDygraph, 'float32', 'float64', 'bool'
)


class TestOperatorOverloadGreaterEqualInDygraph(
    TestOperatorOverloadEqualInDygraph
):
    def run_api(self):
        self.generate_test_value()

        out = self.l_value >= self.r_value
        out_reverse = self.r_value >= self.l_value

        return out, out_reverse


create_test_case(
    TestOperatorOverloadGreaterEqualInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadGreaterEqualInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadGreaterEqualInDygraph, 'float32', 'float64', 'bool'
)


class TestAPIGreaterEqualInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.greater_equal(self.l_value, self.r_value)
        out_reverse = paddle.greater_equal(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIGreaterEqualInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPIGreaterEqualInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPIGreaterEqualInDygraph, 'float32', 'float64', 'bool')


class TestAPIGreaterEqualInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.greater_equal_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.greater_equal_(self.l_value)

        return out, out_reverse


create_test_case(
    TestAPIGreaterEqualInplaceInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestAPIGreaterEqualInplaceInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestAPIGreaterEqualInplaceInDygraph, 'float32', 'float64', 'bool'
)


class TestAPILogicalAndInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.logical_and(self.l_value, self.r_value)
        out_reverse = paddle.logical_and(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPILogicalAndInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILogicalAndInDygraph, 'float32', 'float64', 'bool')

create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'int8', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'int16', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'int32', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'int64', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'float16', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'float32', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'float64', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex64', 'complex128', 'bool')

create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'bool', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'int8', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'int16', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'int32', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'int64', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'float16', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'float32', 'bool')
create_test_case(TestAPILogicalAndInDygraph, 'complex128', 'float64', 'bool')


class TestAPILogicalAndInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.logical_and_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.logical_and_(self.l_value)

        return out, out_reverse


create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'float32', 'float64', 'bool'
)

create_test_case(TestAPILogicalAndInplaceInDygraph, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalAndInplaceInDygraph, 'complex64', 'int8', 'bool')
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex64', 'int16', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex64', 'int32', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex64', 'int64', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex64', 'float16', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex64', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex64', 'float64', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex64', 'complex128', 'bool'
)

create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'bool', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'int8', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'int16', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'int32', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'int64', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'float16', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalAndInplaceInDygraph, 'complex128', 'float64', 'bool'
)


class TestAPILogicalOrInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.logical_or(self.l_value, self.r_value)
        out_reverse = paddle.logical_or(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPILogicalOrInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILogicalOrInDygraph, 'float32', 'float64', 'bool')

create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'int8', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'int16', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'int32', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'int64', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'float16', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'float32', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'float64', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex64', 'complex128', 'bool')

create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'bool', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'int8', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'int16', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'int32', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'int64', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'float16', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'float32', 'bool')
create_test_case(TestAPILogicalOrInDygraph, 'complex128', 'float64', 'bool')


class TestAPILogicalOrInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.logical_or_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.logical_or_(self.l_value)

        return out, out_reverse


create_test_case(TestAPILogicalOrInplaceInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILogicalOrInplaceInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILogicalOrInplaceInDygraph, 'float32', 'float64', 'bool')

create_test_case(TestAPILogicalOrInplaceInDygraph, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalOrInplaceInDygraph, 'complex64', 'int8', 'bool')
create_test_case(TestAPILogicalOrInplaceInDygraph, 'complex64', 'int16', 'bool')
create_test_case(TestAPILogicalOrInplaceInDygraph, 'complex64', 'int32', 'bool')
create_test_case(TestAPILogicalOrInplaceInDygraph, 'complex64', 'int64', 'bool')
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex64', 'float16', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex64', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex64', 'float64', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex64', 'complex128', 'bool'
)

create_test_case(TestAPILogicalOrInplaceInDygraph, 'complex128', 'bool', 'bool')
create_test_case(TestAPILogicalOrInplaceInDygraph, 'complex128', 'int8', 'bool')
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex128', 'int16', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex128', 'int32', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex128', 'int64', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex128', 'float16', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex128', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalOrInplaceInDygraph, 'complex128', 'float64', 'bool'
)


class TestAPILogicalXorInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.logical_xor(self.l_value, self.r_value)
        out_reverse = paddle.logical_xor(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPILogicalXorInDygraph, 'float16', 'float32', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'float16', 'float64', 'bool')

create_test_case(TestAPILogicalXorInDygraph, 'float32', 'float64', 'bool')

create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'int8', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'int16', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'int32', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'int64', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'float16', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'float32', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'float64', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex64', 'complex128', 'bool')

create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'bool', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'int8', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'int16', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'int32', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'int64', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'float16', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'float32', 'bool')
create_test_case(TestAPILogicalXorInDygraph, 'complex128', 'float64', 'bool')


class TestAPILogicalXorInplaceInDygraph(TestOperatorOverloadEqualInDygraph):
    def run_api(self):
        self.generate_test_value()
        out = self.l_value.logical_xor_(self.r_value)

        self.generate_test_value()
        out_reverse = self.r_value.logical_xor_(self.l_value)

        return out, out_reverse


create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'float16', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'float16', 'float64', 'bool'
)

create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'float32', 'float64', 'bool'
)

create_test_case(TestAPILogicalXorInplaceInDygraph, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalXorInplaceInDygraph, 'complex64', 'int8', 'bool')
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex64', 'int16', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex64', 'int32', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex64', 'int64', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex64', 'float16', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex64', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex64', 'float64', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex64', 'complex128', 'bool'
)

create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'bool', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'int8', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'int16', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'int32', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'int64', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'float16', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'float32', 'bool'
)
create_test_case(
    TestAPILogicalXorInplaceInDygraph, 'complex128', 'float64', 'bool'
)


class TestAPIFmaxInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.fmax(self.l_value, self.r_value)
        out_reverse = paddle.fmax(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIFmaxInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIFmaxInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIFmaxInDygraph, 'float32', 'float64', 'float64')


class TestAPIFminInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.fmin(self.l_value, self.r_value)
        out_reverse = paddle.fmin(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIFminInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIFminInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIFminInDygraph, 'float32', 'float64', 'float64')


class TestAPILogAddExpInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.logaddexp(self.l_value, self.r_value)
        out_reverse = paddle.logaddexp(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPILogAddExpInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPILogAddExpInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPILogAddExpInDygraph, 'float32', 'float64', 'float64')


class TestAPIMaximumInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.maximum(self.l_value, self.r_value)
        out_reverse = paddle.maximum(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIMaximumInDygraph, 'float32', 'float64', 'float64')


class TestAPIMinimumInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.minimum(self.l_value, self.r_value)
        out_reverse = paddle.minimum(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIMinimumInDygraph, 'float32', 'float64', 'float64')


class TestAPINextAfterInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.nextafter(self.l_value, self.r_value)
        out_reverse = paddle.nextafter(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPINextAfterInDygraph, 'float32', 'float64', 'float64')


class TestAPIAtan2InDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.atan2(self.l_value, self.r_value)
        out_reverse = paddle.atan2(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIAtan2InDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIAtan2InDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIAtan2InDygraph, 'float32', 'float64', 'float64')


class TestAPIPoissonNllLossInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.nn.functional.poisson_nll_loss(self.l_value, self.r_value)
        out_reverse = paddle.nn.functional.poisson_nll_loss(
            self.r_value, self.l_value
        )

        return out, out_reverse


create_test_case(
    TestAPIPoissonNllLossInDygraph, 'float16', 'float32', 'float32'
)
create_test_case(
    TestAPIPoissonNllLossInDygraph, 'float16', 'float64', 'float64'
)

create_test_case(
    TestAPIPoissonNllLossInDygraph, 'float32', 'float64', 'float64'
)

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPIPoissonNllLossInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPIPoissonNllLossInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPIPoissonNllLossInDygraph, 'bfloat16', 'float64', 'float64'
    )


class TestAPIL1LossInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.nn.functional.l1_loss(self.l_value, self.r_value)
        out_reverse = paddle.nn.functional.l1_loss(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIL1LossInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIL1LossInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIL1LossInDygraph, 'float32', 'float64', 'float64')


class TestAPISmoothL1LossInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.nn.functional.smooth_l1_loss(self.l_value, self.r_value)
        out_reverse = paddle.nn.functional.smooth_l1_loss(
            self.r_value, self.l_value
        )

        return out, out_reverse


create_test_case(TestAPISmoothL1LossInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPISmoothL1LossInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPISmoothL1LossInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPISmoothL1LossInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPISmoothL1LossInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPISmoothL1LossInDygraph, 'bfloat16', 'float64', 'float64'
    )


class TestAPIHuberLossInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle._C_ops.huber_loss(self.l_value, self.r_value, 1.0)
        out_reverse = paddle._C_ops.huber_loss(self.r_value, self.l_value, 1.0)

        return out, out_reverse


create_test_case(TestAPIHuberLossInDygraph, 'float16', 'float32', 'float32')
create_test_case(TestAPIHuberLossInDygraph, 'float16', 'float64', 'float64')

create_test_case(TestAPIHuberLossInDygraph, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPIHuberLossInDygraph, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPIHuberLossInDygraph, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPIHuberLossInDygraph, 'bfloat16', 'float64', 'float64'
    )


class TestAPIMSELossInDygraph(TestOperatorOverloadAddInDygraph):
    def run_api(self):
        self.generate_test_value()

        out = paddle.nn.functional.mse_loss(self.l_value, self.r_value)
        out_reverse = paddle.nn.functional.mse_loss(self.r_value, self.l_value)

        return out, out_reverse


create_test_case(TestAPIMSELossInDygraph, 'float32', 'float64', 'float64')


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
    create_test_case(
        TestOperatorOverloadAddInStatic, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadAddInStatic, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadAddInStatic, 'complex128', 'float64', 'complex128'
)


class TestAPIAddInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.add(self.l_value, self.r_value)
            out_reverse = paddle.add(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIAddInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIAddInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIAddInStatic, 'float32', 'float64', 'float64')


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIAddInStatic, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIAddInStatic, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIAddInStatic, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPIAddInStatic, 'bfloat16', 'complex64', 'complex64')
    create_test_case(TestAPIAddInStatic, 'bfloat16', 'complex128', 'complex128')

create_test_case(TestAPIAddInStatic, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIAddInStatic, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIAddInStatic, 'complex64', 'int64', 'complex64')
create_test_case(TestAPIAddInStatic, 'complex64', 'float16', 'complex64')
create_test_case(TestAPIAddInStatic, 'complex64', 'float32', 'complex64')
create_test_case(TestAPIAddInStatic, 'complex64', 'float64', 'complex128')
create_test_case(TestAPIAddInStatic, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPIAddInStatic, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIAddInStatic, 'complex128', 'int32', 'complex128')
create_test_case(TestAPIAddInStatic, 'complex128', 'int64', 'complex128')
create_test_case(TestAPIAddInStatic, 'complex128', 'float16', 'complex128')
create_test_case(TestAPIAddInStatic, 'complex128', 'float32', 'complex128')
create_test_case(TestAPIAddInStatic, 'complex128', 'float64', 'complex128')


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
    create_test_case(
        TestOperatorOverloadSubInStatic, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadSubInStatic, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadSubInStatic, 'complex128', 'float64', 'complex128'
)


class TestAPISubInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.subtract(self.l_value, self.r_value)
            out_reverse = paddle.subtract(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPISubInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPISubInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPISubInStatic, 'float32', 'float64', 'float64')


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPISubInStatic, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPISubInStatic, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPISubInStatic, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPISubInStatic, 'bfloat16', 'complex64', 'complex64')
    create_test_case(TestAPISubInStatic, 'bfloat16', 'complex128', 'complex128')

create_test_case(TestAPISubInStatic, 'complex64', 'bool', 'complex64')
create_test_case(TestAPISubInStatic, 'complex64', 'int32', 'complex64')
create_test_case(TestAPISubInStatic, 'complex64', 'int64', 'complex64')
create_test_case(TestAPISubInStatic, 'complex64', 'float16', 'complex64')
create_test_case(TestAPISubInStatic, 'complex64', 'float32', 'complex64')
create_test_case(TestAPISubInStatic, 'complex64', 'float64', 'complex128')
create_test_case(TestAPISubInStatic, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPISubInStatic, 'complex128', 'bool', 'complex128')
create_test_case(TestAPISubInStatic, 'complex128', 'int32', 'complex128')
create_test_case(TestAPISubInStatic, 'complex128', 'int64', 'complex128')
create_test_case(TestAPISubInStatic, 'complex128', 'float16', 'complex128')
create_test_case(TestAPISubInStatic, 'complex128', 'float32', 'complex128')
create_test_case(TestAPISubInStatic, 'complex128', 'float64', 'complex128')


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
    create_test_case(
        TestOperatorOverloadMulInStatic, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadMulInStatic, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadMulInStatic, 'complex128', 'float64', 'complex128'
)


class TestAPIMulInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.multiply(self.l_value, self.r_value)
            out_reverse = paddle.multiply(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIMulInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIMulInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIMulInStatic, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIMulInStatic, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIMulInStatic, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIMulInStatic, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPIMulInStatic, 'bfloat16', 'complex64', 'complex64')
    create_test_case(TestAPIMulInStatic, 'bfloat16', 'complex128', 'complex128')

create_test_case(TestAPIMulInStatic, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIMulInStatic, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIMulInStatic, 'complex64', 'int64', 'complex64')
create_test_case(TestAPIMulInStatic, 'complex64', 'float16', 'complex64')
create_test_case(TestAPIMulInStatic, 'complex64', 'float32', 'complex64')
create_test_case(TestAPIMulInStatic, 'complex64', 'float64', 'complex128')
create_test_case(TestAPIMulInStatic, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPIMulInStatic, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIMulInStatic, 'complex128', 'int32', 'complex128')
create_test_case(TestAPIMulInStatic, 'complex128', 'int64', 'complex128')
create_test_case(TestAPIMulInStatic, 'complex128', 'float16', 'complex128')
create_test_case(TestAPIMulInStatic, 'complex128', 'float32', 'complex128')
create_test_case(TestAPIMulInStatic, 'complex128', 'float64', 'complex128')


class TestAPIDivInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.divide(self.l_value, self.r_value)
            out_reverse = paddle.divide(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIDivInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIDivInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIDivInStatic, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIDivInStatic, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIDivInStatic, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIDivInStatic, 'bfloat16', 'float64', 'float64')
    create_test_case(TestAPIDivInStatic, 'bfloat16', 'complex64', 'complex64')
    create_test_case(TestAPIDivInStatic, 'bfloat16', 'complex128', 'complex128')

create_test_case(TestAPIDivInStatic, 'complex64', 'bool', 'complex64')
create_test_case(TestAPIDivInStatic, 'complex64', 'int32', 'complex64')
create_test_case(TestAPIDivInStatic, 'complex64', 'int64', 'complex64')
create_test_case(TestAPIDivInStatic, 'complex64', 'float16', 'complex64')
create_test_case(TestAPIDivInStatic, 'complex64', 'float32', 'complex64')
create_test_case(TestAPIDivInStatic, 'complex64', 'float64', 'complex128')
create_test_case(TestAPIDivInStatic, 'complex64', 'complex128', 'complex128')

create_test_case(TestAPIDivInStatic, 'complex128', 'bool', 'complex128')
create_test_case(TestAPIDivInStatic, 'complex128', 'int32', 'complex128')
create_test_case(TestAPIDivInStatic, 'complex128', 'int64', 'complex128')
create_test_case(TestAPIDivInStatic, 'complex128', 'float16', 'complex128')
create_test_case(TestAPIDivInStatic, 'complex128', 'float32', 'complex128')
create_test_case(TestAPIDivInStatic, 'complex128', 'float64', 'complex128')


class TestOperatorOverloadDivInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value / self.r_value
            out_reverse = self.r_value / self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadDivInStatic, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadDivInStatic, 'float32', 'float64', 'float64'
)


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadDivInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadDivInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadDivInStatic, 'bfloat16', 'float64', 'float64'
    )
    create_test_case(
        TestOperatorOverloadDivInStatic, 'bfloat16', 'complex64', 'complex64'
    )
    create_test_case(
        TestOperatorOverloadDivInStatic, 'bfloat16', 'complex128', 'complex128'
    )

create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'bool', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'int8', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'uint8', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'int16', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'int32', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'int64', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'float16', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'float32', 'complex64'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'float64', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex64', 'complex128', 'complex128'
)

create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'bool', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'int8', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'uint8', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'int16', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'int32', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'int64', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'float16', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'float32', 'complex128'
)
create_test_case(
    TestOperatorOverloadDivInStatic, 'complex128', 'float64', 'complex128'
)


class TestAPIFloorDivInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.floor_divide(self.l_value, self.r_value)
            out_reverse = paddle.floor_divide(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIFloorDivInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIFloorDivInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIFloorDivInStatic, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIFloorDivInStatic, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIFloorDivInStatic, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIFloorDivInStatic, 'bfloat16', 'float64', 'float64')


class TestOperatorOverloadFloorDivInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value // self.r_value
            out_reverse = self.r_value // self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadFloorDivInStatic, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadFloorDivInStatic, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadFloorDivInStatic, 'float32', 'float64', 'float64'
)


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadFloorDivInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadFloorDivInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadFloorDivInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestAPIPowInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.pow(self.l_value, self.r_value)
            out_reverse = paddle.pow(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIPowInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIPowInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIPowInStatic, 'float32', 'float64', 'float64')


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIPowInStatic, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIPowInStatic, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIPowInStatic, 'bfloat16', 'float64', 'float64')


class TestOperatorOverloadPowInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value**self.r_value
            out_reverse = self.r_value**self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadPowInStatic, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadPowInStatic, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadPowInStatic, 'float32', 'float64', 'float64'
)


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadPowInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadPowInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadPowInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestAPIModInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.mod(self.l_value, self.r_value)
            out_reverse = paddle.mod(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIModInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIModInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIModInStatic, 'float32', 'float64', 'float64')


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(TestAPIModInStatic, 'bfloat16', 'float16', 'float32')
    create_test_case(TestAPIModInStatic, 'bfloat16', 'float32', 'float32')
    create_test_case(TestAPIModInStatic, 'bfloat16', 'float64', 'float64')


class TestOperatorOverloadModInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value % self.r_value
            out_reverse = self.r_value % self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadModInStatic, 'float16', 'float32', 'float32'
)
create_test_case(
    TestOperatorOverloadModInStatic, 'float16', 'float64', 'float64'
)

create_test_case(
    TestOperatorOverloadModInStatic, 'float32', 'float64', 'float64'
)


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestOperatorOverloadModInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestOperatorOverloadModInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestOperatorOverloadModInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestOperatorOverloadEqualInStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.ldtype = 'float32'
        self.rdtype = 'float64'
        self.expected_out_dtype = 'bool'

    def generate_test_value(self):
        self.l_value = (paddle.randn((4, 3, 2)) * 10).astype(self.ldtype)
        self.r_value = (paddle.randn((4, 3, 2)) * 10).astype(self.rdtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value == self.r_value
            out_reverse = self.r_value == self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res

    def test_dtype_is_expected(self):
        res = self.run_api()
        self.assertEqual(res[0].dtype.__str__(), self.expected_out_dtype)
        self.assertEqual(res[1].dtype.__str__(), self.expected_out_dtype)


create_test_case(
    TestOperatorOverloadEqualInStatic, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadEqualInStatic, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadEqualInStatic, 'float32', 'float64', 'bool'
)


class TestAPIEqualInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.equal(self.l_value, self.r_value)
            out_reverse = paddle.equal(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIEqualInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPIEqualInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPIEqualInStatic, 'float32', 'float64', 'bool')


class TestAPINotEqualInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.not_equal(self.l_value, self.r_value)
            out_reverse = paddle.not_equal(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPINotEqualInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPINotEqualInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPINotEqualInStatic, 'float32', 'float64', 'bool')


class TestOperatorOverloadNotEqualInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value != self.r_value
            out_reverse = self.r_value != self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadNotEqualInStatic, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadNotEqualInStatic, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadNotEqualInStatic, 'float32', 'float64', 'bool'
)


class TestAPILessThanInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.less_than(self.l_value, self.r_value)
            out_reverse = paddle.less_than(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPILessThanInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPILessThanInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPILessThanInStatic, 'float32', 'float64', 'bool')


class TestOperatorOverloadLessThanInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value < self.r_value
            out_reverse = self.r_value < self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadLessThanInStatic, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadLessThanInStatic, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadLessThanInStatic, 'float32', 'float64', 'bool'
)


class TestAPILessEqualInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.less_equal(self.l_value, self.r_value)
            out_reverse = paddle.less_equal(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPILessEqualInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPILessEqualInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPILessEqualInStatic, 'float32', 'float64', 'bool')


class TestOperatorOverloadLessEqualInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value <= self.r_value
            out_reverse = self.r_value <= self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadLessEqualInStatic, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadLessEqualInStatic, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadLessEqualInStatic, 'float32', 'float64', 'bool'
)


class TestAPIGreaterThanInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.greater_than(self.l_value, self.r_value)
            out_reverse = paddle.greater_than(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIGreaterThanInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPIGreaterThanInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPIGreaterThanInStatic, 'float32', 'float64', 'bool')


class TestOperatorOverloadGreaterThanInStatic(
    TestOperatorOverloadEqualInStatic
):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value > self.r_value
            out_reverse = self.r_value > self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadGreaterThanInStatic, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadGreaterThanInStatic, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadGreaterThanInStatic, 'float32', 'float64', 'bool'
)


class TestAPIGreaterEqualInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.greater_equal(self.l_value, self.r_value)
            out_reverse = paddle.greater_equal(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIGreaterEqualInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPIGreaterEqualInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPIGreaterEqualInStatic, 'float32', 'float64', 'bool')


class TestOperatorOverloadGreaterEqualInStatic(
    TestOperatorOverloadEqualInStatic
):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = self.l_value >= self.r_value
            out_reverse = self.r_value >= self.l_value

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(
    TestOperatorOverloadGreaterEqualInStatic, 'float16', 'float32', 'bool'
)
create_test_case(
    TestOperatorOverloadGreaterEqualInStatic, 'float16', 'float64', 'bool'
)

create_test_case(
    TestOperatorOverloadGreaterEqualInStatic, 'float32', 'float64', 'bool'
)


class TestAPILogicalAndInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.logical_and(self.l_value, self.r_value)
            out_reverse = paddle.logical_and(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPILogicalAndInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPILogicalAndInStatic, 'float32', 'float64', 'bool')

create_test_case(TestAPILogicalAndInStatic, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'int8', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'int16', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'int32', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'int64', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'float16', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'float32', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'float64', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex64', 'complex128', 'bool')

create_test_case(TestAPILogicalAndInStatic, 'complex128', 'bool', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex128', 'int8', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex128', 'int16', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex128', 'int32', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex128', 'int64', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex128', 'float16', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex128', 'float32', 'bool')
create_test_case(TestAPILogicalAndInStatic, 'complex128', 'float64', 'bool')


class TestAPILogicalOrInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.logical_or(self.l_value, self.r_value)
            out_reverse = paddle.logical_or(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPILogicalOrInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPILogicalOrInStatic, 'float32', 'float64', 'bool')

create_test_case(TestAPILogicalOrInStatic, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'int8', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'int16', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'int32', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'int64', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'float16', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'float32', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'float64', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex64', 'complex128', 'bool')

create_test_case(TestAPILogicalOrInStatic, 'complex128', 'bool', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex128', 'int8', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex128', 'int16', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex128', 'int32', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex128', 'int64', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex128', 'float16', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex128', 'float32', 'bool')
create_test_case(TestAPILogicalOrInStatic, 'complex128', 'float64', 'bool')


class TestAPILogicalXorInStatic(TestOperatorOverloadEqualInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.logical_xor(self.l_value, self.r_value)
            out_reverse = paddle.logical_xor(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPILogicalXorInStatic, 'float16', 'float32', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'float16', 'float64', 'bool')

create_test_case(TestAPILogicalXorInStatic, 'float32', 'float64', 'bool')

create_test_case(TestAPILogicalXorInStatic, 'complex64', 'bool', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'int8', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'int16', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'int32', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'int64', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'float16', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'float32', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'float64', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex64', 'complex128', 'bool')

create_test_case(TestAPILogicalXorInStatic, 'complex128', 'bool', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex128', 'int8', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex128', 'int16', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex128', 'int32', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex128', 'int64', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex128', 'float16', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex128', 'float32', 'bool')
create_test_case(TestAPILogicalXorInStatic, 'complex128', 'float64', 'bool')


class TestAPIFmaxInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.fmax(self.l_value, self.r_value)
            out_reverse = paddle.fmax(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIFmaxInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIFmaxInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIFmaxInStatic, 'float32', 'float64', 'float64')


class TestAPIFminInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.fmin(self.l_value, self.r_value)
            out_reverse = paddle.fmin(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIFminInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIFminInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIFminInStatic, 'float32', 'float64', 'float64')


class TestAPILogAddExpInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.logaddexp(self.l_value, self.r_value)
            out_reverse = paddle.logaddexp(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPILogAddExpInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPILogAddExpInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPILogAddExpInStatic, 'float32', 'float64', 'float64')


class TestAPIMaximumInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.maximum(self.l_value, self.r_value)
            out_reverse = paddle.maximum(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIMaximumInStatic, 'float32', 'float64', 'float64')


class TestAPIMinimumInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.minimum(self.l_value, self.r_value)
            out_reverse = paddle.maximum(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIMinimumInStatic, 'float32', 'float64', 'float64')


class TestAPINextAfterInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.nextafter(self.l_value, self.r_value)
            out_reverse = paddle.nextafter(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPINextAfterInStatic, 'float32', 'float64', 'float64')


class TestAPIAtan2InStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.atan2(self.l_value, self.r_value)
            out_reverse = paddle.atan2(self.r_value, self.l_value)

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIAtan2InStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIAtan2InStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIAtan2InStatic, 'float32', 'float64', 'float64')


class TestAPIPoissonNllLossInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.nn.functional.poisson_nll_loss(
                self.l_value, self.r_value
            )
            out_reverse = paddle.nn.functional.poisson_nll_loss(
                self.r_value, self.l_value
            )

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIPoissonNllLossInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIPoissonNllLossInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIPoissonNllLossInStatic, 'float32', 'float64', 'float64')


if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPIPoissonNllLossInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPIPoissonNllLossInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPIPoissonNllLossInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestAPIL1LossInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.nn.functional.l1_loss(self.l_value, self.r_value)
            out_reverse = paddle.nn.functional.l1_loss(
                self.r_value, self.l_value
            )

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIL1LossInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPIL1LossInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPIL1LossInStatic, 'float32', 'float64', 'float64')


class TestAPISmoothL1LossInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.nn.functional.smooth_l1_loss(
                self.l_value, self.r_value
            )
            out_reverse = paddle.nn.functional.smooth_l1_loss(
                self.r_value, self.l_value
            )

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPISmoothL1LossInStatic, 'float16', 'float32', 'float32')
create_test_case(TestAPISmoothL1LossInStatic, 'float16', 'float64', 'float64')

create_test_case(TestAPISmoothL1LossInStatic, 'float32', 'float64', 'float64')

if paddle.is_compiled_with_cuda() and paddle.base.core.supports_bfloat16():
    create_test_case(
        TestAPISmoothL1LossInStatic, 'bfloat16', 'float16', 'float32'
    )
    create_test_case(
        TestAPISmoothL1LossInStatic, 'bfloat16', 'float32', 'float32'
    )
    create_test_case(
        TestAPISmoothL1LossInStatic, 'bfloat16', 'float64', 'float64'
    )


class TestAPIMSELossInStatic(TestOperatorOverloadAddInStatic):
    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()

            out = paddle.nn.functional.mse_loss(self.l_value, self.r_value)
            out_reverse = paddle.nn.functional.mse_loss(
                self.r_value, self.l_value
            )

        res = self.exe.run(prog, fetch_list=[out, out_reverse])
        return res


create_test_case(TestAPIMSELossInStatic, 'float32', 'float64', 'float64')


class TestTypePromotionRaiseError(unittest.TestCase):
    def test_static_type_error(self):
        paddle.enable_static()
        with self.assertRaises(TypeError):
            with paddle.pir_utils.OldIrGuard():
                prog = paddle.static.Program()
                exe = paddle.static.Executor()
                with paddle.static.program_guard(prog):
                    a = paddle.ones([3, 3], dtype='float32')
                    b = paddle.ones([3, 3], dtype='float64')
                    out = a.__matmul__(b)
                    res = exe.run(prog, fetch_list=[out])
                    return res

    def test_dygraph_type_error(self):
        with self.assertRaises(TypeError):
            a = paddle.ones([3, 3], dtype='float32')
            b = paddle.ones([3, 3], dtype='int32')
            return a + b


if __name__ == '__main__':
    unittest.main()
