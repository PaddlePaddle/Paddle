#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest


class TestFetchList(OpTest):
    def setUp(self):
        pass

    def calc_add_out(self, place=None, parallel=None):
        self.x = np.random.random((2, 5)).astype(np.float32)
        self.y = np.random.random((2, 5)).astype(np.float32)
        self.out = np.add(self.x, self.y)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.outputs = {'Out': self.out}
        self.op_type = "elementwise_add"
        self.dtype = np.float32
        outs, fetch_list = self._calc_output(place, parallel=parallel)
        return outs

    def calc_mul_out(self, place=None, parallel=None):
        self.x = np.random.random((2, 5)).astype(np.float32)
        self.y = np.random.random((5, 2)).astype(np.float32)
        self.out = np.dot(self.x, self.y)
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.outputs = {'Out': self.out}
        self.op_type = "elementwise_mul"
        self.dtype = np.float32
        outs, fetch_list = self._calc_output(place, parallel=parallel)
        return outs

    def test_fetch_list(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
                "elementwise_add") and core.op_support_gpu("elementwise_mul"):
            places.append(core.CUDAPlace(0))

        for place in places:
            for parallel in [True, False]:
                add_out = self.calc_add_out(place, parallel)
                add_out1 = np.array(add_out[0])
                mul_out = self.calc_mul_out(place, parallel)
                add_out2 = np.array(add_out[0])
                self.assertTrue(np.array_equal(add_out1, add_out2))


if __name__ == '__main__':
    unittest.main()
