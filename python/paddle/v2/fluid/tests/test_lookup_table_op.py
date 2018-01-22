#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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


class TestLookupTableOp(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        table = np.random.random((17, 31)).astype("float32")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['W'], 'Out', no_grad_set=set('Ids'))


if __name__ == "__main__":
    unittest.main()
