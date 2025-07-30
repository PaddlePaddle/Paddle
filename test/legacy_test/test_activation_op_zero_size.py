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
from test_activation_op import (
    TestAcos,
    TestAcosh,
    TestAsin,
    TestAsinh,
    TestAtan,
    TestAtanh,
    TestCeil,
    TestCELU,
    TestCos,
    TestCosh,
    TestELU,
    TestExpFp32_Prim,
    TestExpm1,
    TestFloor,
    TestHardShrink,
    TestHardSigmoid,
    TestHardSwish,
    TestLeakyRelu,
    TestLogSigmoid,
    TestMish,
    TestReciprocal,
    TestRelu,
    TestRelu6,
    TestRound,
    TestRsqrt,
    TestSigmoid,
    TestSilu,
    TestSin,
    TestSinh,
    TestSoftplus,
    TestSoftshrink,
    TestSoftsign,
    TestSqrt,
    TestSquare,
    TestSTanh,
    TestSwish,
    TestTan,
    TestTanh,
    TestTanhshrink,
    TestThresholdedRelu,
)


# ------------------ Test Zero Size Tensor --------------
def create_test_zero_size_class(parent):
    class TestActZeroSize(parent):
        def init_shape(self):
            self.shape = [12, 0]

        def init_dtype(self):
            self.dtype = np.float64

        def test_check_output(self):
            self.check_output(
                check_pir=True,
                check_symbol_infer=False,
            )

        def test_check_grad(self):
            if self.dtype == np.float16:
                return
            self.check_grad(
                ['X'],
                'Out',
                check_pir=True,
            )

    cls_name = "{}_{}".format(parent.__name__, "ZeroSizeOp")
    TestActZeroSize.__name__ = cls_name
    globals()[cls_name] = TestActZeroSize


create_test_zero_size_class(TestSin)
create_test_zero_size_class(TestCos)
create_test_zero_size_class(TestTan)
create_test_zero_size_class(TestAsin)
create_test_zero_size_class(TestAtan)
create_test_zero_size_class(TestAcos)
create_test_zero_size_class(TestSinh)
create_test_zero_size_class(TestCosh)
create_test_zero_size_class(TestAsinh)
create_test_zero_size_class(TestAcosh)
create_test_zero_size_class(TestAtanh)
create_test_zero_size_class(TestRelu)
create_test_zero_size_class(TestTanh)
create_test_zero_size_class(TestTanhshrink)
create_test_zero_size_class(TestSilu)
create_test_zero_size_class(TestReciprocal)
create_test_zero_size_class(TestSquare)
create_test_zero_size_class(TestSqrt)
create_test_zero_size_class(TestRsqrt)
create_test_zero_size_class(TestSoftsign)
create_test_zero_size_class(TestSigmoid)
create_test_zero_size_class(TestLogSigmoid)
create_test_zero_size_class(TestFloor)
create_test_zero_size_class(TestCeil)
create_test_zero_size_class(TestELU)
create_test_zero_size_class(TestCELU)
create_test_zero_size_class(TestHardShrink)
create_test_zero_size_class(TestHardSigmoid)
create_test_zero_size_class(TestMish)
create_test_zero_size_class(TestSoftplus)
create_test_zero_size_class(TestSoftshrink)
create_test_zero_size_class(TestSTanh)
create_test_zero_size_class(TestThresholdedRelu)
create_test_zero_size_class(TestExpFp32_Prim)
create_test_zero_size_class(TestExpm1)
create_test_zero_size_class(TestLeakyRelu)
create_test_zero_size_class(TestRelu6)
create_test_zero_size_class(TestHardSwish)
create_test_zero_size_class(TestSwish)
create_test_zero_size_class(TestRound)

if __name__ == "__main__":
    unittest.main()
