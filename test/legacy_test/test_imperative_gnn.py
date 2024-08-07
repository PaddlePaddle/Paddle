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

import sys
import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core
from paddle.optimizer import Adam


def gen_data():
    pass


class GraphConv(paddle.nn.Layer):
    def __init__(self, name_scope, in_features, out_features):
        super().__init__(name_scope)

        self._in_features = in_features
        self._out_features = out_features
        self.weight = self.create_parameter(
            attr=None,
            dtype='float32',
            shape=[self._in_features, self._out_features],
        )
        self.bias = self.create_parameter(
            attr=None, dtype='float32', shape=[self._out_features]
        )

    def forward(self, features, adj):
        support = paddle.matmul(features, self.weight)
        # TODO(panyx0718): sparse matmul?
        return paddle.matmul(adj, support) + self.bias


class GCN(paddle.nn.Layer):
    def __init__(self, name_scope, num_hidden):
        super().__init__(name_scope)
        self.gc = GraphConv(self.full_name(), num_hidden, 32)
        self.gc2 = GraphConv(self.full_name(), 32, 10)

    def forward(self, x, adj):
        x = F.relu(self.gc(x, adj))
        return self.gc2(x, adj)


class TestDygraphGNN(unittest.TestCase):
    def test_gnn_float32(self):
        paddle.seed(90)
        paddle.framework.random._manual_program_seed(90)
        startup = base.Program()
        main = base.Program()

        scope = base.core.Scope()
        with new_program_scope(main=main, startup=startup, scope=scope):
            features = paddle.static.data(
                name='features', shape=[1, 100, 50], dtype='float32'
            )
            # Use selected rows when it's supported.
            adj = paddle.static.data(
                name='adj', shape=[1, 100, 100], dtype='float32'
            )
            labels = paddle.static.data(
                name='labels', shape=[100, 1], dtype='int64'
            )

            model = GCN('test_gcn', 50)
            logits = model(features, adj)
            logits = paddle.reshape(logits, logits.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss = paddle.nn.functional.softmax_with_cross_entropy(
                logits, labels
            )
            loss = paddle.sum(loss)

            adam = Adam(learning_rate=1e-3)
            adam.minimize(loss)
            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe.run(startup)
            static_loss = exe.run(
                feed={
                    'features': np.ones([1, 100, 50], dtype=np.float32),
                    'adj': np.ones([1, 100, 100], dtype=np.float32),
                    'labels': np.ones([100, 1], dtype=np.int64),
                },
                fetch_list=[loss],
            )[0]

            static_weight = np.array(
                scope.find_var(model.gc.weight.name).get_tensor()
            )

        with base.dygraph.guard():
            paddle.seed(90)
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(90)

            features = np.ones([1, 100, 50], dtype=np.float32)
            # Use selected rows when it's supported.
            adj = np.ones([1, 100, 100], dtype=np.float32)
            labels = np.ones([100, 1], dtype=np.int64)

            model = GCN('test_gcn', 50)
            logits = model(paddle.to_tensor(features), paddle.to_tensor(adj))
            logits = paddle.reshape(logits, logits.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss = paddle.nn.functional.softmax_with_cross_entropy(
                logits, paddle.to_tensor(labels)
            )
            loss = paddle.sum(loss)
            loss.backward()
            adam = Adam(learning_rate=1e-3, parameters=model.parameters())

            adam.minimize(loss)
            model.clear_gradients()
            loss_value = loss.numpy()
            model_gc_weight_value = model.gc.weight.numpy()

        with base.dygraph.guard():
            paddle.seed(90)
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(), it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(90)

            features2 = np.ones([1, 100, 50], dtype=np.float32)
            # Use selected rows when it's supported.
            adj2 = np.ones([1, 100, 100], dtype=np.float32)
            labels2 = np.ones([100, 1], dtype=np.int64)

            model2 = GCN('test_gcn', 50)
            logits2 = model2(
                paddle.to_tensor(features2), paddle.to_tensor(adj2)
            )
            logits2 = paddle.reshape(logits2, logits2.shape[1:])
            # In other example, it's nll with log_softmax. However, paddle's
            # log_loss only supports binary classification now.
            loss2 = paddle.nn.functional.softmax_with_cross_entropy(
                logits2, paddle.to_tensor(labels2)
            )
            loss2 = paddle.sum(loss2)
            loss2.backward()
            adam2 = Adam(learning_rate=1e-3, parameters=model2.parameters())
            adam2.minimize(loss2)
            model2.clear_gradients()
            loss2_value = loss2.numpy()
            model2_gc_weight_value = model2.gc.weight.numpy()

        self.assertEqual(static_loss, loss_value)
        np.testing.assert_allclose(
            static_weight, model_gc_weight_value, rtol=1e-05
        )
        self.assertEqual(static_loss, loss2_value)
        np.testing.assert_allclose(
            static_weight, model2_gc_weight_value, rtol=1e-05
        )
        sys.stderr.write(f'{static_loss} {loss_value}\n')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
