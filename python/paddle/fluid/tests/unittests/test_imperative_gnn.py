# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest
import numpy as np
import six
import sys

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.imperative.nn import Conv2D, Pool2D, FC
from test_imperative_base import new_program_scope
from paddle.fluid.imperative.base import to_variable


def gen_data():
    pass


class GraphConv(fluid.imperative.Layer):
    def __init__(self, name_scope, in_features, out_features):
        super(GraphConv, self).__init__(name_scope)

        self._in_features = in_features
        self._out_features = out_features
        self.weight = self.create_parameter(
            attr=None,
            dtype='float32',
            shape=[self._in_features, self._out_features])
        self.bias = self.create_parameter(
            attr=None, dtype='float32', shape=[self._out_features])

    def forward(self, features, adj):
        support = fluid.layers.matmul(features, self.weight)
        # TODO(panyx0718): sparse matmul?
        return fluid.layers.matmul(adj, support) + self.bias


class GCN(fluid.imperative.Layer):
    def __init__(self, name_scope, num_hidden):
        super(GCN, self).__init__(name_scope)
        self.gc = GraphConv(self.full_name(), num_hidden, 32)
        self.gc2 = GraphConv(self.full_name(), 32, 10)

    def forward(self, x, adj):
        x = fluid.layers.relu(self.gc(x, adj))
        return self.gc2(x, adj)


class TestImperativeGNN(unittest.TestCase):
    def test_gnn_float32(self):
        seed = 90

        with fluid.imperative.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            features = np.zeros([1, 100, 50], dtype=np.float32)
            adj = np.zeros([1, 100, 100], dtype=np.float32)
            labels = np.zeros([100, 1], dtype=np.int64)

            model = GCN('test_gcn', 50)
            logits = model(to_variable(features), to_variable(adj))
            sys.stderr.write('%s\n' % logits)
            logits = fluid.layers.reshape(logits, logits.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss = fluid.layers.softmax_with_cross_entropy(logits,
                                                           to_variable(labels))
            loss = fluid.layers.reduce_sum(loss)
            sys.stderr.write('%s\n' % loss._numpy())


if __name__ == '__main__':
    unittest.main()
