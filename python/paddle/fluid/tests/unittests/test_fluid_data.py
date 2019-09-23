# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import unittest

from paddle.fluid.data import dimension_is_compatible_with, dtype_is_compatible_with


class TestFluidData(unittest.TestCase):
    def test_data(self):
        x = fluid.data(name='x', shape=[1, 2, 3])
        self.assertTrue(x.desc.need_check_feed())
        y = fluid.layers.data(name='y', shape=[1, 2, 3])
        self.assertFalse(y.desc.need_check_feed())

    def test_dimension_is_compatible_with(self):
        self.assertTrue(dimension_is_compatible_with([1, 2, 3], [1, 2, 3]))
        self.assertTrue(dimension_is_compatible_with([-1, 2, None], [1, 2, 3]))
        self.assertTrue(dimension_is_compatible_with([1, -1, 3], [1, 2, None]))
        self.assertFalse(
            dimension_is_compatible_with([1, -1, 3, 4], [1, 2, None]))
        self.assertFalse(dimension_is_compatible_with([1, 2], [1, 2, 3]))
        self.assertFalse(dimension_is_compatible_with([1, 2, 4], [1, 2, 3]))

    def test_dtype_is_compatible_with(self):
        self.assertTrue(dtype_is_compatible_with('float32', np.float32))
        self.assertFalse(dtype_is_compatible_with(np.int32, 'int64'))
        self.assertFalse(
            dtype_is_compatible_with(core.VarDesc.VarType.UINT8,
                                     core.VarDesc.VarType.INT8))


if __name__ == '__main__':
    unittest.main()
