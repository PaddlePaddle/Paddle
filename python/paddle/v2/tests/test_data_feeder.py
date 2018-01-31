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

    def test_dense(self):
        def compare(input):
            feeder = DataFeeder([('image', data_type.dense_vector(784))],
                                {'image': 0})
            arg = feeder(input)
            output = arg.getSlotValue(0).copyToNumpyMat()
            input = np.array(input, dtype='float32')
            self.assertAlmostEqual(input.all(), output.all())

        # test numpy array
        batch_size = 32
        dim = 784
        data = []
        for i in xrange(batch_size):
            each_sample = []
            each_sample.append(self.dense_reader(dim))
            data.append(each_sample)
        compare(data)

        # each feature is a list
        data = []
        for i in xrange(batch_size):
            each_sample = []
            each_sample.append(self.dense_reader(dim).tolist())
            data.append(each_sample)
        compare(data)

        # test tuple
        data = []
        for i in xrange(batch_size):
            each_sample = (self.dense_reader(dim).tolist(), )
            data.append(each_sample)
        compare(data)

    def test_sparse_binary(self):
        dim = 10000
        batch_size = 32
        data = []
        for i in xrange(batch_size):
            each_sample = []
            each_sample.append(self.sparse_binary_reader(dim, 50))
            data.append(each_sample)
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
            each_sample = []
            a = self.sparse_binary_reader(dim, 40, non_empty=True)
            b = self.dense_reader(len(a)).tolist()
            v.append(a)
            w.append(np.array(b, dtype="float32"))
            each_sample.append(zip(a, b))
            data.append(each_sample)

        feeder = DataFeeder([('input', data_type.sparse_float_vector(dim))],
                            {'input': 0})
        arg = feeder(data)
        output = arg.getSlotValue(0)
        assert isinstance(output, api.Matrix)
        for i in xrange(batch_size):
            self.assertEqual(output.getSparseRowCols(i), v[i])
            cols_value = output.getSparseRowColsVal(i)
            value = [val[1] for val in cols_value]
            value = np.array(value, dtype="float32")
            self.assertAlmostEqual(value.all(), w[i].all())

    def test_integer(self):
        value_range = 100
        batch_size = 32
        index = []
        for i in xrange(batch_size):
            each_sample = []
            each_sample.append(np.random.randint(value_range))
            index.append(each_sample)
        feeder = DataFeeder([('input', data_type.integer_value(value_range))],
                            {'input': 0})
        arg = feeder(index)
        output = arg.getSlotIds(0).copyToNumpyArray()
        index = np.array(index, dtype='int')
        self.assertEqual(output.all(), index.flatten().all())

    def test_integer_sequence(self):
        value_range = 10000
        batch_size = 32
        start = [0]
        data = []
        for i in xrange(batch_size):
            each_sample = []
            each_sample.append(
                self.sparse_binary_reader(
                    value_range, 30, non_empty=True))
            data.append(each_sample)
            start.append(len(each_sample[0]) + start[-1])
        feeder = DataFeeder(
            [('input', data_type.integer_value_sequence(value_range))],
            {'input': 0})
        arg = feeder(data)
        output_data = arg.getSlotIds(0).copyToNumpyArray()
        output_start = arg.getSlotSequenceStartPositions(0).copyToNumpyArray()

        index = []
        for dat in data:
            index.extend(x for x in dat[0])  # only one feature, so dat[0]
        index = np.array(index, dtype='int')
        start = np.array(start, dtype='int')
        self.assertEqual(output_data.all(), index.all())
        self.assertEqual(output_start.all(), start.all())

    def test_multiple_features(self):
        batch_size = 2
        data = []
        for i in xrange(batch_size):
            each_sample = []
            each_sample.append(np.random.randint(10))
            each_sample.append(
                self.sparse_binary_reader(
                    20000, 40, non_empty=True))
            each_sample.append(self.dense_reader(100))
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

        # reader returns 3 features, but only use 2 features
        data_types = [('fea0', data_type.dense_vector(100)),
                      ('fea2', data_type.integer_value(10))]
        feeder = DataFeeder(data_types, {'fea0': 2, 'fea2': 0})
        arg = feeder(data)
        output_dense = arg.getSlotValue(0).copyToNumpyMat()
        output_index = arg.getSlotIds(1).copyToNumpyArray()
        for i in xrange(batch_size):
            self.assertEqual(output_dense[i].all(), data[i][2].all())
            self.assertEqual(output_index[i], data[i][0])

        # reader returns 3 featreus, one is duplicate data
        data_types = [('fea0', data_type.dense_vector(100)),
                      ('fea1', data_type.sparse_binary_vector(20000)),
                      ('fea2', data_type.integer_value(10)),
                      ('fea3', data_type.dense_vector(100))]
        feeder = DataFeeder(data_types,
                            {'fea0': 2,
                             'fea1': 1,
                             'fea2': 0,
                             'fea3': 2})
        arg = feeder(data)
        fea0 = arg.getSlotValue(0).copyToNumpyMat()
        fea1 = arg.getSlotValue(1)
        fea2 = arg.getSlotIds(2).copyToNumpyArray()
        fea3 = arg.getSlotValue(3).copyToNumpyMat()
        for i in xrange(batch_size):
            self.assertEqual(fea0[i].all(), data[i][2].all())
            self.assertEqual(fea1.getSparseRowCols(i), data[i][1])
            self.assertEqual(fea2[i], data[i][0])
            self.assertEqual(fea3[i].all(), data[i][2].all())

    def test_multiple_features_tuple(self):
        batch_size = 2
        data = []
        for i in xrange(batch_size):
            a = np.random.randint(10)
            b = self.sparse_binary_reader(20000, 40, non_empty=True)
            c = self.dense_reader(100)
            each_sample = (a, b, c)
            data.append(each_sample)

        # test multiple features
        data_types = [('fea0', data_type.dense_vector(100)),
                      ('fea1', data_type.sparse_binary_vector(20000)),
                      ('fea2', data_type.integer_value(10))]
        feeder = DataFeeder(data_types, {'fea0': 2, 'fea1': 1, 'fea2': 0})
        arg = feeder(data)
        out_dense = arg.getSlotValue(0).copyToNumpyMat()
        out_sparse = arg.getSlotValue(1)
        out_index = arg.getSlotIds(2).copyToNumpyArray()
        for i in xrange(batch_size):
            self.assertEqual(out_dense[i].all(), data[i][2].all())
            self.assertEqual(out_sparse.getSparseRowCols(i), data[i][1])
            self.assertEqual(out_index[i], data[i][0])

    def test_dense_set_shape(self):
        # test 2-D data
        def gen_data(batch_size, shape):
            data = []
            for i in xrange(batch_size):
                each_sample = []
                each_sample.append(np.random.random(shape))
                data.append(each_sample)
            return data

        feeder = DataFeeder([('image', data_type.dense_array(2352))],
                            {'image': 0})
        arg = feeder(gen_data(32, (3, 28, 28)))
        h = arg.getSlotFrameHeight(0)
        w = arg.getSlotFrameWidth(0)
        self.assertEqual(h, 28)
        self.assertEqual(w, 28)

        arg = feeder(gen_data(32, (3, 30, 32)))
        h = arg.getSlotFrameHeight(0)
        w = arg.getSlotFrameWidth(0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 32)


if __name__ == '__main__':
    api.initPaddle("--use_gpu=0")
    suite = unittest.TestLoader().loadTestsFromTestCase(DataFeederTest)
    unittest.TextTestRunner().run(suite)
    if api.isGpuVersion():
        api.setUseGpu(True)
        unittest.main()
