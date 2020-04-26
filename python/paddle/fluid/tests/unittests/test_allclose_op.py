#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest


class TestAllcloseOp(OpTest):
    def set_args(self):
        self.input = np.array([10000., 1e-07]).astype("float32")
        self.other = np.array([10000.1, 1e-08]).astype("float32")
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False

    def setUp(self):
        self.set_args()
        self.op_type = "allclose"
        self.inputs = {'Input': self.input, 'Other': self.other}
        self.attrs = {
            'rtol': self.rtol,
            'atol': self.atol,
            'equal_nan': self.equal_nan
        }
        self.outputs = {
            'Out': np.array([
                np.allclose(
                    self.inputs['Input'],
                    self.inputs['Other'],
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=self.equal_nan)
            ])
        }

    def test_check_output(self):
        self.check_output()


class TestAllcloseOpSmallNum(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([10000., 1e-08]).astype("float32")
        self.other = np.array([10000.1, 1e-09]).astype("float32")
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False


class TestAllcloseOpNanFalse(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([1.0, float('nan')]).astype("float32")
        self.other = np.array([1.0, float('nan')]).astype("float32")
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = False


class TestAllcloseOpNanTrue(TestAllcloseOp):
    def set_args(self):
        self.input = np.array([1.0, float('nan')]).astype("float32")
        self.other = np.array([1.0, float('nan')]).astype("float32")
        self.rtol = 1e-05
        self.atol = 1e-08
        self.equal_nan = True


if __name__ == "__main__":
    unittest.main()
