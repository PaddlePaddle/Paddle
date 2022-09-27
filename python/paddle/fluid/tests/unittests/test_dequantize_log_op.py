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
import math
from op_test import OpTest


def dequantize_log(x, dict_data):
    output_data = np.zeros_like(x).astype('float32')
    x_f = x.flatten()
    output_data_f = output_data.flatten()
    for i in range(x_f.size):
        if x_f[i] < 0:
            output_data_f[i] = -dict_data[x_f[i] + 128]
        else:
            output_data_f[i] = dict_data[x_f[i]]
    return output_data_f.reshape(x.shape)


class TestDequantizeLogOp(OpTest):

    def setUp(self):
        self.op_type = "dequantize_log"
        x = np.random.randint(low=-128, high=127, size=(20, 10)).astype('int8')
        dict_data = np.random.random(128).astype('float32')
        xdq = dequantize_log(x, dict_data)

        self.inputs = {
            'X': np.array(x).astype('int8'),
            'Dict': np.array(dict_data).astype('float32')
        }
        self.outputs = {'Out': xdq}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
