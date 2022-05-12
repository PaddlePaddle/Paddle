#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy
import collections
from functools import reduce
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.nn.utils import weight_norm, remove_weight_norm


class TestDygraphWeightNorm(unittest.TestCase):
    def setUp(self):
        self.init_test_case()
        self.set_data()

    def init_test_case(self):
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]], )
        self.dim = None

    def set_data(self):
        self.data = collections.OrderedDict()
        for desc in self.data_desc:
            data_name = desc[0]
            data_shape = desc[1]
            data_value = numpy.random.random(
                size=[self.batch_size] + data_shape).astype('float32')
            self.data[data_name] = data_value

    def norm_except_dim(self, w, dim=None):
        shape = w.shape
        ndims = len(shape)
        shape_numel = reduce(lambda x, y: x * y, shape)
        if dim == -1:
            return numpy.linalg.norm(w, axis=None, keepdims=True)
        elif dim == 0:
            tile_shape = list(w.shape)
            tile_shape[0] = 1
            w_matrix = numpy.reshape(w, (shape[0], shape_numel // shape[0]))
            return numpy.linalg.norm(w_matrix, axis=1, keepdims=True)
        elif dim == (ndims - 1):
            w_matrix = numpy.reshape(w, (shape_numel // shape[-1], shape[-1]))
            return numpy.linalg.norm(w_matrix, axis=0, keepdims=True)
        else:
            perm = list(range(ndims))
            perm_ori = list(range(ndims))
            perm[0] = dim
            perm[dim] = 0
            p_transposed = numpy.transpose(w, perm)
            return self.norm_except_dim(p_transposed, 0)

    def weight_normalize(self, w, dim=None):
        shape = w.shape
        ndims = len(shape)
        shape_numel = reduce(lambda x, y: x * y, shape)
        v = w
        g = self.norm_except_dim(w, dim)
        g_mul = g

        if dim == -1:
            v_norm = v / (numpy.linalg.norm(v, axis=None, keepdims=True))
        elif dim == 0:
            w_matrix = numpy.reshape(w, (shape[0], shape_numel // shape[0]))
            v_norm = v / numpy.linalg.norm(w_matrix, axis=1)
            v_norm = numpy.reshape(v_norm, shape)
            g = numpy.squeeze(g, axis=1)
        elif dim == (ndims - 1):
            w_matrix = numpy.reshape(w, (shape_numel // shape[-1], shape[-1]))
            v_norm = v / numpy.linalg.norm(w_matrix, axis=0, keepdims=True)
            v_norm = numpy.reshape(v_norm, shape)
        else:
            perm = list(range(ndims))
            perm[0] = dim
            perm[dim] = 0
            p_transposed = numpy.transpose(v, perm)
            transposed_shape = p_transposed.shape
            transposed_shape_numel = reduce(lambda x, y: x * y,
                                            transposed_shape)
            p_matrix = numpy.reshape(
                p_transposed, (p_transposed.shape[0],
                               transposed_shape_numel // p_transposed.shape[0]))
            v_norm = v / numpy.expand_dims(
                numpy.expand_dims(
                    numpy.linalg.norm(
                        p_matrix, axis=1, keepdims=True), axis=0),
                axis=(ndims - 1))
            v_norm = numpy.reshape(v_norm, transposed_shape)
            v_norm = numpy.transpose(v_norm, perm)
            g = numpy.squeeze(g, axis=1)
            if dim == 1:
                eaxis = 2
            elif dim == 2:
                eaxis = 1
            g_mul = numpy.expand_dims(
                numpy.expand_dims(
                    numpy.expand_dims(
                        g, axis=0), axis=eaxis),
                axis=(ndims - 1))
        w = g_mul * v_norm
        return g, v

    def test_check_output(self):
        fluid.enable_imperative()
        linear = paddle.nn.Conv2D(2, 3, 3)
        before_weight = linear.weight.numpy()
        if self.dim == None:
            self.dim = -1

        if self.dim != -1:
            self.dim = (self.dim + len(before_weight)) % len(before_weight)
        wn = weight_norm(linear, dim=self.dim)
        outputs = []
        for name, data in self.data.items():
            output = linear(fluid.dygraph.to_variable(data))
            outputs.append(output.numpy())
        after_weight = linear.weight
        self.actual_outputs = [linear.weight_g.numpy(), linear.weight_v.numpy()]

        expect_output = self.weight_normalize(before_weight, self.dim)

        for expect, actual in zip(expect_output, self.actual_outputs):
            self.assertTrue(
                numpy.allclose(
                    numpy.array(actual), expect, atol=0.001))


class TestDygraphWeightNormCase1(TestDygraphWeightNorm):
    def init_test_case(self):
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]], )
        self.dim = 0


class TestDygraphWeightNormCase2(TestDygraphWeightNorm):
    def init_test_case(self):
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]], )
        self.dim = 1


class TestDygraphWeightNormCase3(TestDygraphWeightNorm):
    def init_test_case(self):
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]], )
        self.dim = 3


class TestDygraphWeightNormCase4(TestDygraphWeightNorm):
    def init_test_case(self):
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]], )
        self.dim = -3


class TestDygraphRemoveWeightNorm(unittest.TestCase):
    def setUp(self):
        self.init_test_case()

    def init_test_case(self):
        self.batch_size = 3
        self.data_desc = (['x', [2, 3, 3]], )
        self.dim = None

    def test_check_output(self):
        fluid.enable_imperative()
        linear = paddle.nn.Conv2D(2, 3, 3)
        before_weight = linear.weight
        wn = weight_norm(linear, dim=self.dim)
        rwn = remove_weight_norm(linear)
        after_weight = linear.weight
        self.assertTrue(
            numpy.allclose(
                before_weight.numpy(), after_weight.numpy(), atol=0.001))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
