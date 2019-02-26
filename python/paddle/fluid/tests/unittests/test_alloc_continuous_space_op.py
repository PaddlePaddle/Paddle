#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np

from op_test import OpTest


class TestAllocContinuousSpace(OpTest):
    def setUp(self):
        self.op_type = "alloc_continuous_space"
        self.dtype = np.float32
        attrs = self.init_attr()
        self.copy_data = attrs["copy_data"]
        self.constant = attrs["constant"]
        self.set_constant = attrs["set_constant"]
        self.Inputs = self.init_input()
        self.FusedOutput = self.init_output(self.Inputs, self.set_constant,
                                            self.constant)
        self.inputs = {'Input': self.Inputs}
        self.attrs = attrs
        self.outputs = {'Output': self.Inputs, 'FusedOutput': self.FusedOutput}

    def init_dtype(self):
        self.dtype = np.float32

    def init_input(self):
        inputs = []
        inputs.append(("x1", np.random.random([20, 3]).astype(self.dtype)))
        inputs.append(("x2", np.random.random([20]).astype(self.dtype)))
        inputs.append(("x3", np.random.random([1]).astype(self.dtype)))
        inputs.append(("x4", np.random.random([200, 30]).astype(self.dtype)))
        inputs.append(("x5", np.random.random([30]).astype(self.dtype)))
        inputs.append(("x6", np.random.random([1]).astype(self.dtype)))
        return inputs

    def init_attr(self):
        return {"copy_data": True, "set_constant": False, "constant": 0.0}

    def init_output(self, input_list, set_constant, constant):
        inputs = [input[1].flatten() for input in input_list]
        output = np.concatenate(inputs)
        if set_constant:
            output = np.ones((len(output))) * constant
        return output

    def test_check_output(self):
        self.check_output()


class TestAllocContinuousSpace2(TestAllocContinuousSpace):
    def init_attr(self):
        return {"copy_data": False, "set_constant": True, "constant": 0.5}

    def test_check_output(self):
        self.check_output(no_check_set=["Output"])


if __name__ == '__main__':
    unittest.main()
