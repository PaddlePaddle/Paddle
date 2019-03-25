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

import unittest
import numpy as np
import random
import sys

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from test_imperative_base import new_program_scope
from paddle.fluid.imperative.base import to_variable

NUM_USERS = 100
NUM_ITEMS = 1000

BATCH_SIZE = 32
NUM_BATCHES = 2


class MLP(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(MLP, self).__init__(name_scope)
        self._user_latent = fluid.imperative.FC(self.full_name(), 256)
        self._item_latent = fluid.imperative.FC(self.full_name(), 256)

        self._user_layers = []
        self._item_layers = []
        self._hid_sizes = [128, 64]
        for i in range(len(self._hid_sizes)):
            self._user_layers.append(
                self.add_sublayer(
                    'user_layer_%d' % i,
                    fluid.imperative.FC(
                        self.full_name(), self._hid_sizes[i], act='relu')))
            self._item_layers.append(
                self.add_sublayer(
                    'item_layer_%d' % i,
                    fluid.imperative.FC(
                        self.full_name(), self._hid_sizes[i], act='relu')))

    def forward(self, users, items):
        users = self._user_latent(users)
        items = self._item_latent(items)

        for ul, il in zip(self._user_layers, self._item_layers):
            users = ul(users)
            items = il(items)
        return fluid.layers.elementwise_mul(users, items)


class DMF(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(DMF, self).__init__(name_scope)
        self._user_latent = fluid.imperative.FC(self.full_name(), 256)
        self._item_latent = fluid.imperative.FC(self.full_name(), 256)
        self._match_layers = []
        self._hid_sizes = [128, 64]
        for i in range(len(self._hid_sizes)):
            self._match_layers.append(
                self.add_sublayer(
                    'match_layer_%d' % i,
                    fluid.imperative.FC(
                        self.full_name(), self._hid_sizes[i], act='relu')))
        self._mat

    def forward(self, users, items):
        users = self._user_latent(users)
        items = self._item_latent(items)
        match_vec = fluid.layers.concat(
            [users, items], axis=len(users.shape) - 1)
        for l in self._match_layers:
            match_vec = l(match_vec)
        return match_vec


class DeepCF(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(DeepCF, self).__init__(name_scope)

        self._user_emb = fluid.imperative.Embedding(self.full_name(),
                                                    [NUM_USERS, 256])
        self._item_emb = fluid.imperative.Embedding(self.full_name(),
                                                    [NUM_ITEMS, 256])

        self._mlp = MLP(self.full_name())
        self._dmf = DMF(self.full_name())
        self._match_fc = fluid.imperative.FC(self.full_name(), 1, act='sigmoid')

    def forward(self, users, items):
        users_emb = self._user_emb(users)
        items_emb = self._item_emb(items)

        mlp_predictive = self._mlp(users_emb, items_emb)
        dmf_predictive = self._dmf(users_emb, items_emb)
        predictive = fluid.layers.concat(
            [mlp_predictive, dmf_predictive],
            axis=len(mlp_predictive.shape) - 1)
        prediction = self._match_fc(predictive)
        return prediction


def get_data():
    user_ids = []
    item_ids = []
    labels = []
    for uid in range(NUM_USERS):
        for iid in range(NUM_ITEMS):
            # 10% positive
            label = float(random.randint(1, 10) == 1)
            user_ids.append(uid)
            item_ids.append(iid)
            labels.append(label)
    indices = np.arange(NUM_USERS * NUM_ITEMS)
    np.random.shuffle(indices)
    users_np = np.array(user_ids, dtype=np.int64)[indices]
    items_np = np.array(item_ids, dtype=np.int64)[indices]
    labels_np = np.array(labels, dtype=np.float32)[indices]
    return np.expand_dims(users_np, -1), \
           np.expand_dims(items_np, -1), \
           np.expand_dims(labels_np, -1)


class TestImperativeDeepCF(unittest.TestCase):
    def test_gan_float32(self):
        seed = 90
        users_np, items_np, labels_np = get_data()

        startup = fluid.Program()
        startup.random_seed = seed
        main = fluid.Program()
        main.random_seed = seed

        scope = fluid.core.Scope()
        with new_program_scope(main=main, startup=startup, scope=scope):
            users = fluid.layers.data('users', [1], dtype='int64')
            items = fluid.layers.data('items', [1], dtype='int64')
            labels = fluid.layers.data('labels', [1], dtype='float32')

            deepcf = DeepCF('deepcf')
            prediction = deepcf(users, items)
            loss = fluid.layers.reduce_sum(
                fluid.layers.log_loss(prediction, labels))
            adam = fluid.optimizer.AdamOptimizer(0.01)
            adam.minimize(loss)

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(startup)
            for slice in range(0, BATCH_SIZE * NUM_BATCHES, BATCH_SIZE):
                static_loss = exe.run(
                    main,
                    feed={
                        users.name: users_np[slice:slice + BATCH_SIZE],
                        items.name: items_np[slice:slice + BATCH_SIZE],
                        labels.name: labels_np[slice:slice + BATCH_SIZE]
                    },
                    fetch_list=[loss])[0]
                sys.stderr.write('static loss %s\n' % static_loss)

        with fluid.imperative.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            deepcf = DeepCF('deepcf')
            for slice in range(0, BATCH_SIZE * NUM_BATCHES, BATCH_SIZE):
                prediction = deepcf(
                    to_variable(users_np[slice:slice + BATCH_SIZE]),
                    to_variable(items_np[slice:slice + BATCH_SIZE]))
                loss = fluid.layers.reduce_sum(
                    fluid.layers.log_loss(prediction,
                                          to_variable(labels_np[slice:slice +
                                                                BATCH_SIZE])))
                loss._backward()
                adam = fluid.optimizer.AdamOptimizer(0.01)
                adam.minimize(loss)
                deepcf.clear_gradients()
                dy_loss = loss._numpy()

        self.assertEqual(static_loss, dy_loss)


if __name__ == '__main__':
    unittest.main()
