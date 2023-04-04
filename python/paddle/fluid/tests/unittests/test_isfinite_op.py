# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from eager_op_test import OpTest, convert_float_to_uint16

from paddle.fluid import core


class TestInf(OpTest):
    def setUp(self):
        self.op_type = "isinf"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        x[0] = np.inf
        x[-1] = np.inf

        self.inputs = {'X': x}
        self.outputs = {'Out': np.array(True).astype(self.dtype)}

    def init_dtype(self):
        pass

    def test_output(self):
        self.check_output()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16Inf(TestInf):
    def init_dtype(self):
        self.dtype = np.float16


# BFP16 isinf Test
class TestInfBF16(OpTest):
    def setUp(self):
        self.op_type = "isinf"
        self.dtype = np.uint16
        x = np.random.uniform(0.1, 1, [11, 17]).astype(np.float32)
        x[0] = np.inf
        x[-1] = np.inf

        out = np.isinf(x).astype(np.float32)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_output(self):
        self.check_output()


class TestNAN(OpTest):
    def setUp(self):
        self.op_type = "isnan"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        x[0] = np.nan
        x[-1] = np.nan

        self.inputs = {'X': x}
        self.outputs = {'Out': np.array(True).astype(self.dtype)}

    def init_dtype(self):
        pass

    def test_output(self):
        self.check_output()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16NAN(TestNAN):
    def init_dtype(self):
        self.dtype = np.float16


# BFP16 isnan Test
class TestNANBF16(OpTest):
    def setUp(self):
        self.op_type = "isnan"
        self.dtype = np.uint16
        x = np.random.uniform(0.1, 1, [11, 17]).astype(np.float32)
        x[0] = np.nan
        x[-1] = np.nan

        out = np.isnan(x).astype(np.float32)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_output(self):
        self.check_output()


class TestIsfinite(OpTest):
    def setUp(self):
        self.op_type = "isfinite"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        x[0] = np.inf
        x[-1] = np.nan
        out = np.isinf(x) | np.isnan(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': np.array(False).astype(self.dtype)}

    def init_dtype(self):
        pass

    def test_output(self):
        self.check_output()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFP16Isfinite(TestIsfinite):
    def init_dtype(self):
        self.dtype = np.float16


# BFP16 isfinite Test
class TestIsfiniteBF16(OpTest):
    def setUp(self):
        self.op_type = "isfinite"
        self.dtype = np.uint16
        x = np.random.uniform(0.1, 1, [11, 17]).astype(np.float32)
        x[0] = np.inf
        x[-1] = np.nan

        out = np.isfinite(x).astype(np.float32)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
