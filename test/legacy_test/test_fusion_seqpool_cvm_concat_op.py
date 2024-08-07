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

import sys
import unittest

import numpy as np
from op_test import OpTest

import paddle

sys.path.append("../sequence")
from test_cvm_op import cvm_compute
from test_sequence_pool import (
    compute_seqpool_avg,
    compute_seqpool_sqrt,
    compute_seqpool_sum,
)


def convert_to_offset(lod):
    offset = [[0] for i in lod]
    for i, level in enumerate(lod):
        for seq_len in level:
            offset[i].append(offset[i][-1] + seq_len)
    return offset


def api_wrapper(x, cvm, pooltype="SUM", use_cvm=True, axis=1):
    if isinstance(x, paddle.Tensor):
        x = [x]
    return paddle._C_ops.fusion_seqpool_cvm_concat(
        x, cvm, pooltype, use_cvm, axis
    )


class TestFusionSeqPoolCVMConcatOp(OpTest):
    def setUp(self):
        self.w = 11
        self.use_cvm = True
        self.lods = [[[2, 3, 5]], [[1, 5, 2]]]
        self.set_conf()
        self.set_pooltype()
        self.op_type = 'fusion_seqpool_cvm_concat'
        self.python_api = api_wrapper
        self.axis = 1
        bs = len(self.lods[0][0])
        inputs = []
        outs = []
        # The cvm variable is not actually used.
        cvm = np.array([[0.6, 0.4]]).astype("float32")
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
                out = cvm_compute(out, self.w, self.use_cvm)
            elif self.pooltype == "AVERAGE":
                compute_seqpool_avg(x, offset, out)
                out = cvm_compute(out, self.w, self.use_cvm)
            elif self.pooltype == "SQRT":
                compute_seqpool_sqrt(x, offset, out)
                out = cvm_compute(out, self.w, self.use_cvm)
            else:
                raise Exception("Unsupported pool type!")
            inputs.append((f'x_{i}', (x, lod)))
            outs.append(out)
            i = i + 1

        self.inputs = {'X': inputs, "CVM": cvm}
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


class TestFusionSeqPoolCVMConcatOpCase1(TestFusionSeqPoolCVMConcatOp):
    def set_conf(self):
        self.lods = [[[1]]]


class TestFusionSeqPoolCVMConcatOpCase2(TestFusionSeqPoolCVMConcatOp):
    def set_conf(self):
        self.lods = [[[1]], [[1]], [[1]]]


class TestFusionSeqPoolCVMConcatOpCase3(TestFusionSeqPoolCVMConcatOp):
    def set_conf(self):
        self.lods = [[[1, 3, 4, 6]]]
        self.w = 10


class TestFusionSeqPoolCVMConcatOpCase4(TestFusionSeqPoolCVMConcatOp):
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

    cls_name_avg = "{}_{}".format(parent.__name__, "avg")
    cls_name_sqrt = "{}_{}".format(parent.__name__, "sqrt")
    TestSeqPoolAvgCase.__name__ = cls_name_avg
    TestSeqPoolSqrtCase.__name__ = cls_name_sqrt
    globals()[cls_name_avg] = TestSeqPoolAvgCase
    globals()[cls_name_sqrt] = TestSeqPoolSqrtCase


create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOp)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase1)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase2)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase3)
create_test_avg_sqrt_class(TestFusionSeqPoolCVMConcatOpCase4)

if __name__ == '__main__':
    unittest.main()
