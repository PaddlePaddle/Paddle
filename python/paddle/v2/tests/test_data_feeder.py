# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import py_paddle.swig_paddle as api
import numpy as np

from paddle.v2 import data_type
from paddle.v2.data_feeder import DataFeeder


class DataFeederTest(unittest.TestCase):
    def dense_reader(self, size):
        data = np.random.random(size)
        return data

    def sparse_binary_reader(self, high, size_limit, non_empty=False):
        num = np.random.randint(size_limit)  # num could be 0
        while non_empty and num == 0:
            num = np.random.randint(size_limit)
        return np.random.randint(high, size=num).tolist()

    def test_dense_vector(self):
        def compare(input):
            feeder = DataFeeder([('image', data_type.dense_vector(784))],
                                {'image': 0})
            arg = feeder([input])
            output = arg.getSlotValue(0).copyToNumpyMat()
            input = np.array(input, dtype='float32')
            self.assertAlmostEqual(input.all(), output.all())

        # test numpy array
        batch_size = 32
        dim = 784
        data = []
        for i in xrange(batch_size):
            data.append(self.dense_reader(784))
        compare(data)

        # test list
        data = []
        for i in xrange(batch_size):
            data.append(self.dense_reader(784).tolist())
        compare(data)

    def test_sparse_binary(self):
        dim = 10000
        batch_size = 32
        data = []
        for i in xrange(batch_size):
            data.append([self.sparse_binary_reader(dim, 50)])
        feeder = DataFeeder([('input', data_type.sparse_binary_vector(dim))],
                            {'input': 0})
        arg = feeder(data)
        output = arg.getSlotValue(0)
        assert isinstance(output, api.Matrix)
        for i in xrange(batch_size):
            self.assertEqual(output.getSparseRowCols(i), data[i][0])

    def test_sparse(self):
        dim = 10000
        batch_size = 32
        v = []
        w = []
        data = []
        for dat in xrange(batch_size):
            a = self.sparse_binary_reader(dim, 40, non_empty=True)
            b = self.dense_reader(len(a)).tolist()
            v.append(a)
            w.append(b[0])
            data.append([zip(a, b)])

        feeder = DataFeeder([('input', data_type.sparse_vector(dim))],
                            {'input': 0})
        arg = feeder(data)
        output = arg.getSlotValue(0)
        assert isinstance(output, api.Matrix)
        for i in xrange(batch_size):
            self.assertEqual(output.getSparseRowCols(i), v[i])

    def test_integer(self):
        dim = 100
        batch_size = 32
        index = []
        for i in xrange(batch_size):
            index.append([np.random.randint(dim)])
        feeder = DataFeeder([('input', data_type.integer_value(dim))],
                            {'input': 0})
        arg = feeder(index)
        output = arg.getSlotIds(0).copyToNumpyArray()
        index = np.array(index, dtype='int')
        self.assertEqual(output.all(), index.flatten().all())

    def test_multiple_slots(self):
        batch_size = 2
        data = []
        for i in xrange(batch_size):
            each_sample = []
            each_sample.append(np.random.randint(10))  # size of feature 2: 10
            each_sample.append(
                self.sparse_binary_reader(
                    20000, 40, non_empty=True))  # size of feature 1: 20000
            each_sample.append(self.dense_reader(100))  # size of feature 0: 100
            data.append(each_sample)

        # test multiple features
        data_types = [('fea0', data_type.dense_vector(100)),
                      ('fea1', data_type.sparse_binary_vector(20000)),
                      ('fea2', data_type.integer_value(10))]
        feeder = DataFeeder(data_types, {'fea0': 2, 'fea1': 1, 'fea2': 0})
        arg = feeder(data)
        output_dense = arg.getSlotValue(0).copyToNumpyMat()
        output_sparse = arg.getSlotValue(1)
        output_index = arg.getSlotIds(2).copyToNumpyArray()
        for i in xrange(batch_size):
            self.assertEqual(output_dense[i].all(), data[i][2].all())
            self.assertEqual(output_sparse.getSparseRowCols(i), data[i][1])
            self.assertEqual(output_index[i], data[i][0])

        # reader returns 3 featreus, but only use 2 features
        data_types = [('fea0', data_type.dense_vector(100)),
                      ('fea2', data_type.integer_value(10))]
        feeder = DataFeeder(data_types, {'fea0': 2, 'fea2': 0})
        arg = feeder(data)
        output_dense = arg.getSlotValue(0).copyToNumpyMat()
        output_index = arg.getSlotIds(1).copyToNumpyArray()
        for i in xrange(batch_size):
            self.assertEqual(output_dense[i].all(), data[i][2].all())
            self.assertEqual(output_index[i], data[i][0])


if __name__ == '__main__':
    api.initPaddle("--use_gpu=0")
    unittest.main()

if __name__ == '__main__':
    api.initPaddle("--use_gpu=0")
    unittest.main()
