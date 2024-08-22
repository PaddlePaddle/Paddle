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

import struct
import unittest

import numpy as np
from op_test import OpTest

import paddle


def api_wrapper(w, ids, padding_idx=-1):
    return paddle._C_ops.lookup_table_dequant(w, ids, padding_idx)


class TestLookupTableDequantOp(OpTest):
    def setUp(self):
        self.op_type = "lookup_table_dequant"
        self.python_api = api_wrapper
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
