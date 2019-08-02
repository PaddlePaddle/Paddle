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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
from test_reorder_lod_tensor import convert_to_offset
from test_seq_pool import compute_seqpool_sum, compute_seqpool_avg, compute_seqpool_sqrt
from test_cvm_op import cvm_compute


class TestFusionSeqPoolCVMConcatOp(OpTest):
    """This class is used to test FusionSeqPoolCVMConcatOp."""

    def setUp(self):
        """Constructing an initialization environment."""
        self.w = 11
        self.use_cvm = True
        self.lod = [[1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5]]
        self.set_conf()
        self.set_pooltype()
        self.op_type = 'fusion_seqpool_cvm_concat'
        self.slots_num = 2
        self.axis = 1
        bs = len(self.lod[0]) / self.slots_num
        inputs = []
        # The cvm variable is not actually used.
        cvm = np.array([[0.6, 0.4]]).astype("float32")

        assert len(self.lod[0]) % self.slots_num == 0
        x = np.random.uniform(0.1, 1,
                              [sum(self.lod[0]), self.w]).astype('float32')
        offset = convert_to_offset(self.lod)
        out = np.zeros((bs * self.slots_num, self.w)).astype('float32')
        if self.pooltype == "SUM":
            compute_seqpool_sum(x, offset, out)
            out = cvm_compute(out, self.w, self.use_cvm)
        elif self.pooltype == "AVERAGE":
            compute_seqpool_avg(x, offset, out)
            out = cvm_compute(out, self.w, self.use_cvm)
        elif self.pooltype == "SQRT":
            compute_seqpool_sqrt(x, offset, out)
            out = cvm_compute(out, self.w, self.use_cvm)
        else:
            raise Exception("Unsupported pool type!")
        inputs.append(('x', (x, self.lod)))

        self.inputs = {'X': inputs, "CVM": cvm}
        self.outputs = {'Out': out}
        self.attrs = {
            'pooltype': self.pooltype,
            'axis': self.axis,
            'slots_num': self.slots_num,
        }

    def set_pooltype(self):
        self.pooltype = "SUM"

    def set_conf(self):
        pass

    def test_check_output(self):
        self.check_output()


def create_test_avg_sqrt_class(parent):
    """Test averge and square root behavior."""

    class TestSeqPoolAvgCase(parent):
        def set_pooltype(self):
            self.pooltype = "AVERAGE"

    class TestSeqPoolSqrtCase(parent):
        def set_pooltype(self):
            self.pooltype = "SQRT"

    cls_name_avg = "{0}_{1}".format(parent.__name__, "avg")
    cls_name_sqrt = "{0}_{1}".format(parent.__name__, "sqrt")
    TestSeqPoolAvgCase.__name__ = cls_name_avg
    TestSeqPoolSqrtCase.__name__ = cls_name_sqrt
    globals()[cls_name_avg] = TestSeqPoolAvgCase
    globals()[cls_name_sqrt] = TestSeqPoolSqrtCase


create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOp)

if __name__ == '__main__':
    unittest.main()
