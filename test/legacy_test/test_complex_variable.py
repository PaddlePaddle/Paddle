#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.base.dygraph as dg
from paddle.base import core
from paddle.base.data_feeder import convert_dtype
from paddle.base.framework import convert_np_dtype_to_dtype_


class TestComplexVariable(unittest.TestCase):
    def compare(self):
        a = np.array(
            [[1.0 + 1.0j, 2.0 + 1.0j], [3.0 + 1.0j, 4.0 + 1.0j]]
        ).astype(self._dtype)
        b = np.array([[1.0 + 1.0j, 1.0 + 1.0j]]).astype(self._dtype)

        with dg.guard():
            x = paddle.to_tensor(a)
            y = paddle.to_tensor(b)
            out = paddle.add(x, y)

        np.testing.assert_allclose(out.numpy(), a + b, rtol=1e-05)
        self.assertEqual(out.dtype, convert_np_dtype_to_dtype_(self._dtype))
        self.assertEqual(out.shape, x.shape)

    def test_attrs(self):
        self._dtype = "complex64"
        self.compare()
        self._dtype = "complex128"
        self.compare()

    def test_convert_np_dtype_to_dtype(self):
        if paddle.framework.use_pir_api():
            self.assertEqual(
                convert_np_dtype_to_dtype_(np.complex64),
                core.DataType.COMPLEX64,
            )
            self.assertEqual(
                convert_np_dtype_to_dtype_(np.complex64),
                core.DataType.COMPLEX64,
            )
        else:
            self.assertEqual(
                convert_np_dtype_to_dtype_(np.complex64),
                core.VarDesc.VarType.COMPLEX64,
            )
            self.assertEqual(
                convert_np_dtype_to_dtype_(np.complex64),
                core.VarDesc.VarType.COMPLEX64,
            )

    def test_convert_dtype(self):
        self.assertEqual(
            convert_dtype(core.VarDesc.VarType.COMPLEX64), "complex64"
        )
        self.assertEqual(
            convert_dtype(core.VarDesc.VarType.COMPLEX128), "complex128"
        )


if __name__ == '__main__':
    unittest.main()
