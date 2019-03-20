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


def compute_seqpool_sum(x, offset, out):
    for i in range(len(offset[0]) - 1):
        sub_x = x[offset[0][i]:offset[0][i + 1], :]
        out[i] = sub_x.sum(axis=0)


def compute_seqpool_avg(x, offset, out):
    for i in range(len(offset[0]) - 1):
        sub_x = x[offset[0][i]:offset[0][i + 1], :]
        out[i] = sub_x.mean(axis=0)


def compute_seqpool_sqrt(x, offset, out):
    for i in range(len(offset[0]) - 1):
        sub_x = x[offset[0][i]:offset[0][i + 1], :]
        seq_len = offset[0][i + 1] - offset[0][i]
        out[i] = sub_x.sum(axis=0) / np.sqrt(seq_len)


class TestSeqAvgPool(OpTest):
    def set_data(self):
        self.op_type = 'sequence_pool'
        # one level, batch size is 4
        x = np.random.uniform(0.1, 1, [11, 23]).astype('float32')
        lod = [[11]]
        self.inputs = {'X': (x, lod)}
        offset = convert_to_offset(lod)
        out = np.zeros((len(lod[0]), 23)).astype('float32')
        self.outputs = {'Out': out}
        return x, offset, out

    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "AVERAGE"}
        compute_seqpool_avg(x, offset, out)

    def setUp(self):
        x, offset, out = self.set_data()
        self.compute(x, offset, out)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        # Remove MaxIndex after check_grad is refined.
        self.outputs['MaxIndex'] = \
            np.zeros(self.outputs['Out'].shape).astype('int32')
        self.check_grad(["X"], "Out")


class TestSeqSumPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "SUM"}
        compute_seqpool_sum(x, offset, out)


class TestSeqMaxPool(TestSeqAvgPool):
    def set_data(self):
        self.op_type = 'sequence_pool'
        x = np.random.uniform(0.1, 1, [13, 23]).astype('float32')
        lod = [[13]]
        offset = convert_to_offset(lod)
        for i in range(len(offset[0]) - 1):
            l = offset[0][i + 1] - offset[0][i]
            x[offset[0][i] + np.random.randint(l), :] += 2.0

        self.inputs = {'X': (x, lod)}

        out = np.zeros((1, 23)).astype('float32')
        self.outputs = {'Out': out}
        return x, offset, out

    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "MAX"}
        for i in range(len(offset[0]) - 1):
            sub_x = x[offset[0][i]:offset[0][i + 1], :]
            out[i] = np.amax(sub_x, axis=0)


class TestSeqSqrtPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "SQRT"}
        compute_seqpool_sqrt(x, offset, out)


class TestSeqLastPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "LAST"}
        for i in range(len(offset[0]) - 1):
            sub_x = x[offset[0][i]:offset[0][i + 1], :]
            out[i] = sub_x[-1, :]


class TestSeqFirstPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "FIRST"}
        for i in range(len(offset[0]) - 1):
            sub_x = x[offset[0][i]:offset[0][i + 1], :]
            out[i] = sub_x[0, :]


class TestSeqAvgPool2D(TestSeqAvgPool):
    def set_data(self):
        self.op_type = 'sequence_pool'
        # one level, batch size is 4
        x = np.random.uniform(0.1, 1, [13, 3, 17]).astype('float32')
        lod = [[4, 1, 3, 5]]
        self.inputs = {'X': (x, lod)}
        offset = convert_to_offset(lod)

        out = np.zeros((4, 3, 17)).astype('float32')
        self.outputs = {'Out': out}
        return x, offset, out

    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "AVERAGE"}
        for i in range(len(offset[0]) - 1):
            sub_x = np.reshape(x[offset[0][i]:offset[0][i + 1], :],
                               (-1, 3 * 17))
            out[i] = np.reshape(sub_x.mean(axis=0), (3, 17))


class TestSeqSumPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "SUM"}
        for i in range(len(offset[0]) - 1):
            sub_x = np.reshape(x[offset[0][i]:offset[0][i + 1], :],
                               (-1, 3 * 17))
            out[i] = np.reshape(sub_x.sum(axis=0), (3, 17))


class TestSeqSqrtPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "SQRT"}
        for i in range(len(offset[0]) - 1):
            sub_x = np.reshape(x[offset[0][i]:offset[0][i + 1], :],
                               (-1, 3 * 17))
            seq_len = offset[0][i + 1] - offset[0][i]
            out[i] = np.reshape(sub_x.sum(axis=0) / np.sqrt(seq_len), (3, 17))

    def test_check_grad(self):
        # Remove MaxIndex after check_grad is refined.
        self.outputs['MaxIndex'] = \
            np.zeros(self.outputs['Out'].shape).astype('int32')
        self.check_grad(["X"], "Out", max_relative_error=0.06)


class TestSeqMaxPool2D(TestSeqAvgPool2D):
    def set_data(self):
        self.op_type = 'sequence_pool'
        x = np.random.uniform(0.1, 1, [13, 3, 11]).astype('float32')
        lod = [[4, 1, 3, 5]]
        self.inputs = {'X': (x, lod)}
        offset = convert_to_offset(lod)
        for i in range(len(offset[0]) - 1):
            l = offset[0][i + 1] - offset[0][i]
            x[offset[0][i] + np.random.randint(l), :] += 1.0

        out = np.zeros((4, 3, 11)).astype('float32')
        self.outputs = {'Out': out}
        return x, offset, out

    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "MAX"}
        for i in range(len(offset[0]) - 1):
            sub_x = np.reshape(x[offset[0][i]:offset[0][i + 1], :],
                               (-1, 3 * 11))
            out[i] = np.reshape(np.amax(sub_x, axis=0), (3, 11))


class TestSeqMaxPool2DInference(TestSeqMaxPool2D):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "MAX", 'is_test': True}
        for i in range(len(offset[0]) - 1):
            sub_x = np.reshape(x[offset[0][i]:offset[0][i + 1], :],
                               (-1, 3 * 11))
            out[i] = np.reshape(np.amax(sub_x, axis=0), (3, 11))

    def test_check_grad(self):
        """Grad computation does not apply to Sequence MAX 
            Pool executed when is_test is true """
        return


class TestSeqLastPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "LAST"}
        for i in range(len(offset[0]) - 1):
            sub_x = np.reshape(x[offset[0][i]:offset[0][i + 1], :],
                               (-1, 3 * 17))
            out[i] = np.reshape(sub_x[-1, :], (3, 17))


class TestSeqFirstPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {'pooltype': "FIRST"}
        for i in range(len(offset[0]) - 1):
            sub_x = np.reshape(x[offset[0][i]:offset[0][i + 1], :],
                               (-1, 3 * 17))
            out[i] = np.reshape(sub_x[0, :], (3, 17))


if __name__ == '__main__':
    unittest.main()
