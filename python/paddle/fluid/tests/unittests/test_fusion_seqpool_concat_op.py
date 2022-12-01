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
from test_reorder_lod_tensor import convert_to_offset
from sequence.test_sequence_pool import (
    compute_seqpool_sum,
    compute_seqpool_avg,
    compute_seqpool_sqrt,
)


class TestFusionSeqPoolConcatOp(OpTest):
    def setUp(self):
        self.w = 11
        self.lods = [[[2, 3, 5]], [[1, 5, 2]]]
        self.set_conf()
        self.set_pooltype()
        self.op_type = 'fusion_seqpool_concat'
        self.axis = 1
        bs = len(self.lods[0][0])
        inputs = []
        outs = []
        i = 0
        for lod in self.lods:
            assert bs == len(lod[0]), 'All lod size should be equal'
            x = np.random.uniform(0.1, 1, [sum(lod[0]), self.w]).astype(
                'float32'
            )
            offset = convert_to_offset(lod)
            out = np.zeros((bs, self.w)).astype('float32')
            if self.pooltype == "SUM":
                compute_seqpool_sum(x, offset, out)
            elif self.pooltype == "AVERAGE":
                compute_seqpool_avg(x, offset, out)
            elif self.pooltype == "SQRT":
                compute_seqpool_sqrt(x, offset, out)
            else:
                raise Exception("Unsupported pool type!")
            inputs.append(('x_{0}'.format(i), (x, lod)))
            outs.append(out)
            i = i + 1

        self.inputs = {'X': inputs}
        self.outputs = {'Out': np.concatenate(outs, axis=self.axis)}
        self.attrs = {
            'pooltype': self.pooltype,
            'axis': self.axis,
        }

    def set_pooltype(self):
        self.pooltype = "SUM"

    def set_conf(self):
        pass

    def test_check_output(self):
        self.check_output()


class TestFusionSeqPoolConcatOpCase1(TestFusionSeqPoolConcatOp):
    def set_conf(self):
        self.lods = [[[1]]]


class TestFusionSeqPoolConcatOpCase2(TestFusionSeqPoolConcatOp):
    def set_conf(self):
        self.lods = [[[1]], [[1]], [[1]]]


class TestFusionSeqPoolConcatOpCase3(TestFusionSeqPoolConcatOp):
    def set_conf(self):
        self.lods = [[[1, 3, 4, 6]]]
        self.w = 10


class TestFusionSeqPoolConcatOpCase4(TestFusionSeqPoolConcatOp):
    def set_conf(self):
        self.lods = [[[2, 13, 4]], [[1, 1, 1]], [[5, 3, 1]], [[9, 10, 3]]]
        self.w = 3


# test avg pool and sqrt
def create_test_avg_sqrt_class(parent):
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


create_test_avg_sqrt_class(TestFusionSeqPoolConcatOp)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase1)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase2)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase3)
create_test_avg_sqrt_class(TestFusionSeqPoolConcatOpCase4)

if __name__ == '__main__':
    unittest.main()
