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
import sys

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.optimizer import AdamOptimizer
from test_imperative_base import new_program_scope
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.framework import _test_eager_guard


def gen_data():
    pass


class GraphConv(fluid.Layer):
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


class GCN(fluid.Layer):
    def __init__(self, name_scope, num_hidden):
        super(GCN, self).__init__(name_scope)
        self.gc = GraphConv(self.full_name(), num_hidden, 32)
        self.gc2 = GraphConv(self.full_name(), 32, 10)

    def forward(self, x, adj):
        x = fluid.layers.relu(self.gc(x, adj))
        return self.gc2(x, adj)


class TestDygraphGNN(unittest.TestCase):
    def func_gnn_float32(self):
        paddle.seed(90)
        paddle.framework.random._manual_program_seed(90)
        startup = fluid.Program()
        main = fluid.Program()

        scope = fluid.core.Scope()
        with new_program_scope(main=main, startup=startup, scope=scope):
            features = fluid.layers.data(
                name='features',
                shape=[1, 100, 50],
                dtype='float32',
                append_batch_size=False)
            # Use selected rows when it's supported.
            adj = fluid.layers.data(
                name='adj',
                shape=[1, 100, 100],
                dtype='float32',
                append_batch_size=False)
            labels = fluid.layers.data(
                name='labels',
                shape=[100, 1],
                dtype='int64',
                append_batch_size=False)

            model = GCN('test_gcn', 50)
            logits = model(features, adj)
            logits = fluid.layers.reshape(logits, logits.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss = fluid.layers.softmax_with_cross_entropy(logits, labels)
            loss = fluid.layers.reduce_sum(loss)

            adam = AdamOptimizer(learning_rate=1e-3)
            adam.minimize(loss)
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(startup)
            static_loss = exe.run(feed={
                'features': np.ones(
                    [1, 100, 50], dtype=np.float32),
                'adj': np.ones(
                    [1, 100, 100], dtype=np.float32),
                'labels': np.ones(
                    [100, 1], dtype=np.int64)
            },
                                  fetch_list=[loss])[0]

            static_weight = np.array(
                scope.find_var(model.gc.weight.name).get_tensor())

        with fluid.dygraph.guard():
            paddle.seed(90)
            paddle.framework.random._manual_program_seed(90)

            features = np.ones([1, 100, 50], dtype=np.float32)
            # Use selected rows when it's supported.
            adj = np.ones([1, 100, 100], dtype=np.float32)
            labels = np.ones([100, 1], dtype=np.int64)

            model = GCN('test_gcn', 50)
            logits = model(to_variable(features), to_variable(adj))
            logits = fluid.layers.reshape(logits, logits.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss = fluid.layers.softmax_with_cross_entropy(logits,
                                                           to_variable(labels))
            loss = fluid.layers.reduce_sum(loss)
            loss.backward()
            adam = AdamOptimizer(
                learning_rate=1e-3, parameter_list=model.parameters())

            adam.minimize(loss)
            model.clear_gradients()
            loss_value = loss.numpy()
            model_gc_weight_value = model.gc.weight.numpy()

        with fluid.dygraph.guard():
            paddle.seed(90)
            paddle.framework.random._manual_program_seed(90)

            features2 = np.ones([1, 100, 50], dtype=np.float32)
            # Use selected rows when it's supported.
            adj2 = np.ones([1, 100, 100], dtype=np.float32)
            labels2 = np.ones([100, 1], dtype=np.int64)

            model2 = GCN('test_gcn', 50)
            logits2 = model2(to_variable(features2), to_variable(adj2))
            logits2 = fluid.layers.reshape(logits2, logits2.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss2 = fluid.layers.softmax_with_cross_entropy(
                logits2, to_variable(labels2))
            loss2 = fluid.layers.reduce_sum(loss2)
            loss2.backward()
            adam2 = AdamOptimizer(
                learning_rate=1e-3, parameter_list=model2.parameters())
            adam2.minimize(loss2)
            model2.clear_gradients()
            loss2_value = loss2.numpy()
            model2_gc_weight_value = model2.gc.weight.numpy()

        self.assertEqual(static_loss, loss_value)
        self.assertTrue(np.allclose(static_weight, model_gc_weight_value))
        self.assertEqual(static_loss, loss2_value)
        self.assertTrue(np.allclose(static_weight, model2_gc_weight_value))
        sys.stderr.write('%s %s\n' % (static_loss, loss_value))

    def test_gnn_float32(self):
        with _test_eager_guard():
            self.func_gnn_float32()
        self.func_gnn_float32()


if __name__ == '__main__':
    unittest.main()
