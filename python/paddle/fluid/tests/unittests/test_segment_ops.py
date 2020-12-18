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


def compute_segment_min_max(x, segment_ids, pooltype="MAX"):
    length = segment_ids[-1] + 1
    target_shape = list(x.shape)
    target_shape[0] = length
    gradient = np.zeros_like(x)
    results = np.zeros(target_shape, dtype=x.dtype)
    last_idx = 0
    current_id = segment_ids[0]
    for idx in range(1, len(segment_ids) + 1):
        if idx < len(segment_ids):
            if segment_ids[idx] == current_id:
                continue
        sub_x = x[last_idx:idx, :]
        if pooltype == "MAX":
            results[current_id] = np.amax(sub_x, axis=0)
        elif pooltype == "MIN":
            results[current_id] = np.amin(sub_x, axis=0)
        else:
            raise ValueError("Invalid pooltype, only MAX, MIN supported!")
        gradient[last_idx:idx, :][sub_x == results[current_id]] = 1
        last_idx = idx
        if idx < len(segment_ids):
            current_id = segment_ids[idx]

    return results, gradient / results.size


class TestSegmentOps(OpTest):
    def set_data(self):
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        segment_ids = self.set_segment(len(x), len(x) // 5 + 1)
        return x, segment_ids

    def set_segment(self, origin_len, reduce_len):
        segment = np.zeros(reduce_len, dtype='int64')
        segment = np.random.randint(0, reduce_len, size=[origin_len])
        segment = np.sort(segment)
        return segment.astype('int64')

    def compute(self, x, segment_ids):
        return compute_segment_sum(x, segment_ids)

    def prepare(self):
        self.op_type = "segment_pool"
        self.dtype = np.float64
        self.shape = [30, 15]
        self.attrs = {"pooltype": "SUM"}

    def setUp(self):
        self.prepare()
        x, segment_ids = self.set_data()
        result = self.compute(x, segment_ids)
        self.inputs = {
            'X': x.astype(self.dtype),
            'SegmentIds': segment_ids.astype(np.int64)
        }
        self.outputs = {'Out': result.astype(self.dtype)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")


class TestSegmentSum2(TestSegmentOps):
    def prepare(self):
        super(TestSegmentSum2, self).prepare()
        self.shape = [40, 20]
        self.dtype = np.float32

    def setUp(self):
        self.prepare()
        x, segment_ids = self.set_data()
        result = self.compute(x, segment_ids)
        self.inputs = {
            'X': x.astype(self.dtype),
            'SegmentIds': segment_ids.astype(np.int32)
        }
        self.outputs = {'Out': result.astype(self.dtype)}


class TestSegmentMax(TestSegmentOps):
    def compute(self, x, segment_ids):
        return compute_segment_min_max(x, segment_ids, pooltype="MAX")

    def prepare(self):
        super(TestSegmentMax, self).prepare()
        self.shape = [40, 20]
        self.attrs = {'pooltype': "MAX"}

    def setUp(self):
        self.prepare()
        x, segment_ids = self.set_data()
        result, self.gradient = self.compute(x, segment_ids)
        self.inputs = {
            'X': x.astype(self.dtype),
            'SegmentIds': segment_ids.astype(np.int32)
        }
        self.outputs = {'Out': result.astype(self.dtype)}

    def test_check_grad(self):
        self.check_grad(["X"], "Out", user_defined_grads=[self.gradient])


class TestSegmentMax2(TestSegmentMax):
    def prepare(self):
        super(TestSegmentMax2, self).prepare()
        self.dtype = np.float32


class TestSegmentMin(TestSegmentMax):
    def compute(self, x, segment_ids):
        return compute_segment_min_max(x, segment_ids, pooltype="MIN")

    def prepare(self):
        super(TestSegmentMin, self).prepare()
        self.attrs = {'pooltype': "MIN"}


class TestSegmentMin2(TestSegmentMin):
    def prepare(self):
        super(TestSegmentMin2, self).prepare()
        self.dtype = np.float32


class TestSegmentMean(TestSegmentOps):
    def compute(self, x, segment_ids):
        return compute_segment_mean(x, segment_ids)

    def prepare(self):
        super(TestSegmentMean, self).prepare()
        self.shape = [40, 20]
        self.attrs = {'pooltype': "MEAN"}

    def setUp(self):
        self.prepare()
        x, segment_ids = self.set_data()
        result = self.compute(x, segment_ids)
        self.inputs = {'X': x, 'SegmentIds': segment_ids}
        self.outputs = {
            'Out': result,
            'SummedIds': compute_segment_sum(
                np.ones([len(x), 1]).astype(self.dtype), segment_ids)
        }


class TestSegmentMean2(TestSegmentMean):
    def prepare(self):
        super(TestSegmentMean2, self).prepare()
        self.dtype = np.float32
        self.shape = [30, 20]
        self.attrs = {'pooltype': "MEAN"}


if __name__ == '__main__':
    unittest.main()
