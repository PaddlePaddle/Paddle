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
import paddle.fluid.core as core
import numpy as np
from op_test import OpTest


class TestScaleOp(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.dtype = np.float32
        self.init_dtype()

        self.inputs = {'X': np.random.random((2, 3)).astype(self.dtype)}
        self.attrs = {'scale': -2.3}
        self.outputs = {'Out': self.inputs['X'] * self.attrs['scale']}

    def init_dtype(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


# NOTE(dzhwinter): for the float16 gradient tests, the round error
# is larger. You should set a larger margin for atol, max_relative_error
class TestScaleFP16Op(TestScaleOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output(atol=2e-1)

    def test_check_grad(self):
        self.check_grad(set(["X"]), "Out", max_relative_error=0.02)


if __name__ == "__main__":
    unittest.main()
