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

import unittest, sys
sys.path.append("../")
import numpy as np
from test_reduce_op import TestSumOp, Test1DReduce, \
     Test2DReduce0, Test2DReduce1, Test3DReduce0, Test3DReduce1, Test3DReduce2, \
     Test3DReduce3, TestKeepDimReduce, TestKeepDimReduceSumMultiAxises, \
     TestReduceSumWithDimOne, TestReduceSumWithNumelOne


class Test3DReduce21(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [1, 2]}
        self.inputs = {'X': np.random.random((20, 1, 5)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


if __name__ == '__main__':
    unittest.main()
