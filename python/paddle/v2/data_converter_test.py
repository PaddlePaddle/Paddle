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
import paddle.trainer.PyDataProvider2 as dp2

from paddle.v2.data_converter import DataConverter


class DataConverterTest(unittest.TestCase):
    def dense_reader(self, shape):
        data = np.random.random(shape)
        return data

    def sparse_binary_reader(self,
                             high,
                             size_limit,
                             batch_size,
                             non_empty=False):
        data = []
        for i in xrange(batch_size):
            num = np.random.randint(size_limit)  # num could be 0
            while non_empty and num == 0:
                num = np.random.randint(size_limit)
            data.append(np.random.randint(high, size=num).tolist())

        return data

    def test_dense_vector(self):
        def compare(input):
            converter = DataConverter([('image', dp2.dense_vector(784))])
            arg = converter([input], {'image': 0})
            output = arg.getSlotValue(0).copyToNumpyMat()
            input = np.array(input, dtype='float32')
            self.assertAlmostEqual(input.all(), output.all())

        # test numpy array
        data = self.dense_reader(shape=[32, 784])
        compare(data)

        # test list
        compare(data.tolist())

    #def test_sparse_binary(self):
    #    dim = 100000
    #    data = self.sparse_binary_reader(dim, 5, 2)
    #    converter = DataConverter([('input', dp2.sparse_binary_vector(dim))])
    #    arg = converter([data], {'input':0})
    #    output = arg.getSlotValue(0)

    #def test_sparse(self):
    #    dim = 100000
    #    v = self.sparse_binary_reader(dim, 5, 2)
    #    w = []
    #    for dat in data:
    #        x = self.dense_reader(shape=[1, len(dat)])
    #        w.append(x.tolist())
    #    data = []
    #    for each in zip(v, w):
    #        data.append(zip(each[0], each[1]))
    #    
    #    converter = DataConverter([('input', dp2.sparse_binary_vector(dim))])
    #    arg = converter([data], {'input':0})
    #    output = arg.getSlotValue(0)

    def test_integer(self):
        dim = 100
        index = np.random.randint(dim, size=32)
        print index
        converter = DataConverter([('input', dp2.integer_value(dim))])
        arg = converter([index], {'input': 0})
        print arg.getSlotValue(0)
        output = arg.getSlotValue(0).copyToNumpyArray()
        print 'output=', output


if __name__ == '__main__':
    unittest.main()
