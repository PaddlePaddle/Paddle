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


class TestRefByTrainerIdOp(OpTest):
    def setUp(self):
        self.op_type = "ref_by_trainer_id"
        param_baks = [("x%d" % x, np.random.random((10, 10)).astype("float32"))
                      for x in range(10)]
        self.inputs = {
            'X': param_baks,
            'TrainerId': np.array([8]).astype("int64")
        }
        self.outputs = {'Out': param_baks[8][1]}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
