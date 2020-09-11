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
import sys
from op_test import OpTest

#, skip_check_grad_ci
#from test_reorder_lod_tensor import convert_to_offset


def compute_segment_sum(x, segment_ids):
    length = segment_ids[-1] + 1
    target_shape = list(x.shape)
    target_shape[0] = length
    results = np.zeros(target_shape, dtype=x.dtype)
    for index, ids in enumerate(segment_ids):
        results[ids, :] += x[index, :]
    return results


def compute_segment_mean(x, segment_ids):
    length = segment_ids[-1] + 1
    target_shape = list(x.shape)
    target_shape[0] = length
    results = np.zeros(target_shape, dtype=x.dtype)
    count = np.zeros(length, dtype=x.dtype) + 1e-8
    for index, ids in enumerate(segment_ids):
        results[ids, :] += x[index, :]
        count[ids] += 1
    results = results / count.reshape([-1, 1])
    return results


class TestSegmentOps(OpTest):
    def set_data(self):
        x = np.random.uniform(0.1, 1, [30, 15]).astype('float32')
        return x

    def set_segment(self, origin_len, reduce_len):
        segment = np.zeros(reduce_len, dtype='int64')
        segment = np.random.randint(0, reduce_len, size=[origin_len])
        segment = np.sort(segment)
        return segment.astype('int64')

    def compute(self, x, segment_ids):
        return compute_segment_sum(x, segment_ids)

    def setUp(self):
        x = self.set_data()
        self.dtype = np.float64
        segment_ids = self.set_segment(len(x), len(x) // 5 + 1)
        self.op_type = "segment_sum"
        result = self.compute(x, segment_ids)
        self.inputs = {
            'X': x.astype(self.dtype),
            'SegmentIds': segment_ids.astype(np.int64)
        }
        self.attrs = {}
        self.outputs = {'Out': result.astype(self.dtype)}

    def test_check_output(self):
        #self.check_output(check_dygraph=False)
        self.check_output()

    def test_check_grad(self):
        #self.check_grad(["X"], "Out", check_dygraph=False)
        self.check_grad(["X"], "Out")


class TestSegmentSum2(TestSegmentOps):
    def setUp(self):
        x = self.set_data()
        self.dtype = np.float32
        segment_ids = self.set_segment(len(x), len(x) // 5 + 1)
        self.op_type = "segment_sum"
        result = self.compute(x, segment_ids)
        self.inputs = {
            'X': x.astype(self.dtype),
            'SegmentIds': segment_ids.astype(np.int32)
        }
        self.attrs = {}
        self.outputs = {'Out': result.astype(self.dtype)}


#class TestSegmentMean(OpTest):
#    def set_data(self):
#        x = np.random.uniform(0.1, 1, [11, 23]).astype('float32')
#        return x
#
#    def set_segment(self, origin_len, reduce_len):
#        segment = np.zeros(reduce_len, dtype='int64')
#        segment = np.random.randint(0, reduce_len,  size=[origin_len])
#        segment = np.sort(segment)
#        return segment
#
#    def compute(self, x, segment_ids):
#        return compute_segment_mean(x, segment_ids)
#
#    def setUp(self):
#        x = self.set_data()
#        segment_ids = self.set_segment(len(x), len(x)//5+1)
#        self.op_type = "segment_mean"
#        result = self.compute(x, segment_ids)
#        self.inputs = {'X':x, 'SegmentIds':segment_ids}
#        self.attrs = {}
#        self.outputs = {'Out': result}
#
#    def test_check_output(self):
#        self.check_output()
#
#    def test_check_grad(self):
#        self.check_grad(["X"], "Out")
#

if __name__ == '__main__':
    unittest.main()
