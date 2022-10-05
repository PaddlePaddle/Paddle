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
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.compat as cpt
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import struct


class TestLookupTableDequantOp(OpTest):

    def setUp(self):
        self.op_type = "lookup_table_dequant"
        table = np.random.random((17, 32)).astype("float32")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}

        # calculate output
        output = []
        for id in ids:
            tmp = []
            min, max = table[id][0], table[id][1]
            for val in table[id][2:]:
                tmp += [
                    int(x) * (max - min) / pow(2, 8) + min
                    for x in bytearray(struct.pack("f", val))
                ]
            output.append(tmp)

        self.outputs = {'Out': np.asarray(output, dtype="float32")}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
