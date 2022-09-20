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
import sys

import numpy as np
import paddle
import paddle.fluid.core as core

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


def segment_pool_split(X, SegmentIds, pooltype):
    if pooltype == "SUM":
        return paddle.geometric.segment_sum(X, SegmentIds)
    elif pooltype == "MEAN":
        return paddle.geometric.segment_mean(X, SegmentIds)
    elif pooltype == "MIN":
        return paddle.geometric.segment_min(X, SegmentIds)
    elif pooltype == "MAX":
        return paddle.geometric.segment_max(X, SegmentIds)


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
        self.python_api = segment_pool_split
        self.python_out_sig = ["Out"]
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
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(["X"], "Out", check_eager=True)


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
            'Out':
            result,
            'SummedIds':
            compute_segment_sum(
                np.ones([len(x), 1]).astype(self.dtype), segment_ids)
        }


class TestSegmentMean2(TestSegmentMean):

    def prepare(self):
        super(TestSegmentMean2, self).prepare()
        self.dtype = np.float32
        self.shape = [30, 20]
        self.attrs = {'pooltype': "MEAN"}


class API_SegmentOpsTest(unittest.TestCase):

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            y = paddle.static.data(name='y', shape=[3], dtype='int32')

            res_sum = paddle.incubate.segment_sum(x, y)
            res_mean = paddle.incubate.segment_mean(x, y)
            res_max = paddle.incubate.segment_max(x, y)
            res_min = paddle.incubate.segment_min(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            data2 = np.array([0, 0, 1], dtype="int32")

            np_sum = np.array([[4, 4, 4], [4, 5, 6]], dtype="float32")
            np_mean = np.array([[2, 2, 2], [4, 5, 6]], dtype="float32")
            np_max = np.array([[3, 2, 3], [4, 5, 6]], dtype="float32")
            np_min = np.array([[1, 2, 1], [4, 5, 6]], dtype="float32")

            ret = exe.run(feed={
                'x': data1,
                'y': data2
            },
                          fetch_list=[res_sum, res_mean, res_max, res_min])

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_dygraph(self):
        device = paddle.CPUPlace()
        with paddle.fluid.dygraph.guard(device):
            x = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]],
                                 dtype='float32')
            y = paddle.to_tensor([0, 0, 1], dtype="int32")
            res_sum = paddle.incubate.segment_sum(x, y)
            res_mean = paddle.incubate.segment_mean(x, y)
            res_max = paddle.incubate.segment_max(x, y)
            res_min = paddle.incubate.segment_min(x, y)

            np_sum = np.array([[4, 4, 4], [4, 5, 6]], dtype="float32")
            np_mean = np.array([[2, 2, 2], [4, 5, 6]], dtype="float32")
            np_max = np.array([[3, 2, 3], [4, 5, 6]], dtype="float32")
            np_min = np.array([[1, 2, 1], [4, 5, 6]], dtype="float32")

            ret = [res_sum, res_mean, res_max, res_min]

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res,
                                       ret_res.numpy(),
                                       rtol=1e-05,
                                       atol=1e-06)


class API_GeometricSegmentOpsTest(unittest.TestCase):

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name="x", shape=[3, 3], dtype="float32")
            y = paddle.static.data(name='y', shape=[3], dtype='int32')

            res_sum = paddle.geometric.segment_sum(x, y)
            res_mean = paddle.geometric.segment_mean(x, y)
            res_max = paddle.geometric.segment_max(x, y)
            res_min = paddle.geometric.segment_min(x, y)

            exe = paddle.static.Executor(paddle.CPUPlace())
            data1 = np.array([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
            data2 = np.array([0, 0, 1], dtype="int32")

            np_sum = np.array([[4, 4, 4], [4, 5, 6]], dtype="float32")
            np_mean = np.array([[2, 2, 2], [4, 5, 6]], dtype="float32")
            np_max = np.array([[3, 2, 3], [4, 5, 6]], dtype="float32")
            np_min = np.array([[1, 2, 1], [4, 5, 6]], dtype="float32")

            ret = exe.run(feed={
                'x': data1,
                'y': data2
            },
                          fetch_list=[res_sum, res_mean, res_max, res_min])

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res, ret_res, rtol=1e-05, atol=1e-06)

    def test_dygraph(self):
        device = paddle.CPUPlace()
        with paddle.fluid.dygraph.guard(device):
            x = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]],
                                 dtype='float32')
            y = paddle.to_tensor([0, 0, 1], dtype="int32")
            res_sum = paddle.geometric.segment_sum(x, y)
            res_mean = paddle.geometric.segment_mean(x, y)
            res_max = paddle.geometric.segment_max(x, y)
            res_min = paddle.geometric.segment_min(x, y)

            np_sum = np.array([[4, 4, 4], [4, 5, 6]], dtype="float32")
            np_mean = np.array([[2, 2, 2], [4, 5, 6]], dtype="float32")
            np_max = np.array([[3, 2, 3], [4, 5, 6]], dtype="float32")
            np_min = np.array([[1, 2, 1], [4, 5, 6]], dtype="float32")

            ret = [res_sum, res_mean, res_max, res_min]

        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res,
                                       ret_res.numpy(),
                                       rtol=1e-05,
                                       atol=1e-06)

    def test_dygraph_cpu_float16(self):
        device = paddle.CPUPlace()
        with paddle.fluid.dygraph.guard(device):
            x = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]],
                                 dtype='float16')
            y = paddle.to_tensor([0, 0, 1], dtype="int32")
            res_sum = paddle.geometric.segment_sum(x, y)
            res_mean = paddle.geometric.segment_mean(x, y)
            res_max = paddle.geometric.segment_max(x, y)
            res_min = paddle.geometric.segment_min(x, y)

            np_sum = np.array([[4, 4, 4], [4, 5, 6]], dtype="float16")
            np_mean = np.array([[2, 2, 2], [4, 5, 6]], dtype="float16")
            np_max = np.array([[3, 2, 3], [4, 5, 6]], dtype="float16")
            np_min = np.array([[1, 2, 1], [4, 5, 6]], dtype="float16")

            ret = [res_sum, res_mean, res_max, res_min]
        for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
            np.testing.assert_allclose(np_res,
                                       ret_res.numpy(),
                                       rtol=1e-05,
                                       atol=1e-06)

    def test_dygraph_cuda_float16(self):
        if core.is_compiled_with_cuda():
            device = paddle.CUDAPlace(0)
            with paddle.fluid.dygraph.guard(device):
                x = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]],
                                     dtype='float16')
                y = paddle.to_tensor([0, 0, 1], dtype="int32")
                res_sum = paddle.geometric.segment_sum(x, y)
                res_mean = paddle.geometric.segment_mean(x, y)
                res_max = paddle.geometric.segment_max(x, y)
                res_min = paddle.geometric.segment_min(x, y)

                np_sum = np.array([[4, 4, 4], [4, 5, 6]], dtype="float16")
                np_mean = np.array([[2, 2, 2], [4, 5, 6]], dtype="float16")
                np_max = np.array([[3, 2, 3], [4, 5, 6]], dtype="float16")
                np_min = np.array([[1, 2, 1], [4, 5, 6]], dtype="float16")

                ret = [res_sum, res_mean, res_max, res_min]

            for np_res, ret_res in zip([np_sum, np_mean, np_max, np_min], ret):
                np.testing.assert_allclose(np_res,
                                           ret_res.numpy(),
                                           rtol=1e-05,
                                           atol=1e-06)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
