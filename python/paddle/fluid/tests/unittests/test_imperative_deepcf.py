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
import os
import sys

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from test_imperative_base import new_program_scope
from paddle.fluid.imperative.base import to_variable

DATA_PATH = os.environ.get('DATA_PATH', '')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 256))
NUM_BATCHES = int(os.environ.get('NUM_BATCHES', 2))
NUM_EPOCHES = int(os.environ.get('NUM_EPOCHES', 1))


class DMF(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(DMF, self).__init__(name_scope)
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


class MLP(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(MLP, self).__init__(name_scope)
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
    def __init__(self, name_scope, num_users, num_items, matrix):
        super(DeepCF, self).__init__(name_scope)
        self._num_users = num_users
        self._num_items = num_items
        self._rating_matrix = self.create_parameter(
            None,
            matrix.shape,
            matrix.dtype,
            is_bias=False,
            default_initializer=fluid.initializer.NumpyArrayInitializer(matrix))
        self._rating_matrix._stop_gradient = True

        # self._user_emb = fluid.imperative.Embedding(self.full_name(),
        #                                             [self._num_users, 256])
        # self._item_emb = fluid.imperative.Embedding(self.full_name(),
        #                                             [self._num_items, 256])

        self._mlp = MLP(self.full_name())
        self._dmf = DMF(self.full_name())
        self._match_fc = fluid.imperative.FC(self.full_name(), 1, act='sigmoid')

    def forward(self, users, items):
        # users_emb = self._user_emb(users)
        # items_emb = self._item_emb(items)
        sys.stderr.write('forward: %s\n' % users._stop_gradient)
        users_emb = fluid.layers.gather(self._rating_matrix, users)
        items_emb = fluid.layers.gather(
            fluid.layers.transpose(self._rating_matrix, [1, 0]), items)
        users_emb.stop_gradient = True
        items_emb.stop_gradient = True

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
    matrix = np.zeros([100, 1000], dtype=np.float32)

    NUM_USERS = 100
    NUM_ITEMS = 1000
    for uid in range(NUM_USERS):
        for iid in range(NUM_ITEMS):
            label = float(random.randint(1, 6) == 1)
            user_ids.append(uid)
            item_ids.append(iid)
            labels.append(label)
            matrix[uid, iid] = label
    indices = np.arange(len(user_ids))
    np.random.shuffle(indices)
    users_np = np.array(user_ids, dtype=np.int32)[indices]
    items_np = np.array(item_ids, dtype=np.int32)[indices]
    labels_np = np.array(labels, dtype=np.float32)[indices]
    return np.expand_dims(users_np, -1), \
           np.expand_dims(items_np, -1), \
           np.expand_dims(labels_np, -1), NUM_USERS, NUM_ITEMS, matrix


def load_data(DATA_PATH):
    sys.stderr.write('loading from %s\n' % DATA_PATH)
    likes = dict()
    num_users = -1
    num_items = -1
    with open(DATA_PATH, 'r') as f:
        for l in f.readlines():
            uid, iid, rating = [int(v) for v in l.split('\t')]
            num_users = max(num_users, uid + 1)
            num_items = max(num_items, iid + 1)
            if float(rating) > 0.0:
                likes[(uid, iid)] = 1.0

    user_ids = []
    item_ids = []
    labels = []
    matrix = np.zeros([num_users, num_items], dtype=np.float32)
    for uid, iid in likes.keys():
        user_ids.append(uid)
        item_ids.append(iid)
        labels.append(1.0)
        matrix[uid, iid] = 1.0

        negative = 0
        while negative < 3:
            nuid = random.randint(0, num_users - 1)
            niid = random.randint(0, num_items - 1)
            if (nuid, niid) not in likes:
                negative += 1
                user_ids.append(nuid)
                item_ids.append(niid)
                labels.append(0.0)

    indices = np.arange(len(user_ids))
    np.random.shuffle(indices)
    users_np = np.array(user_ids, dtype=np.int32)[indices]
    items_np = np.array(item_ids, dtype=np.int32)[indices]
    labels_np = np.array(labels, dtype=np.float32)[indices]
    return np.expand_dims(users_np, -1), \
           np.expand_dims(items_np, -1), \
           np.expand_dims(labels_np, -1), num_users, num_items, matrix


class TestImperativeDeepCF(unittest.TestCase):
    def test_deefcf(self):
        seed = 90
        if DATA_PATH:
            (users_np, items_np, labels_np, num_users, num_items,
             matrix) = load_data(DATA_PATH)
        else:
            (users_np, items_np, labels_np, num_users, num_items,
             matrix) = get_data()

        startup = fluid.Program()
        startup.random_seed = seed
        main = fluid.Program()
        main.random_seed = seed
        """
        scope = fluid.core.Scope()
        with new_program_scope(main=main, startup=startup, scope=scope):
            users = fluid.layers.data('users', [1], dtype='int32')
            items = fluid.layers.data('items', [1], dtype='int32')
            labels = fluid.layers.data('labels', [1], dtype='float32')

            deepcf = DeepCF('deepcf', num_users, num_items, matrix)
            prediction = deepcf(users, items)
            loss = fluid.layers.reduce_sum(
                fluid.layers.log_loss(prediction, labels))
            adam = fluid.optimizer.AdamOptimizer(0.01)
            adam.minimize(loss)

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(startup)
            for e in range(NUM_EPOCHES):
                sys.stderr.write('epoch %d\n' % e)
                for slice in range(0, BATCH_SIZE * NUM_BATCHES, BATCH_SIZE):
                    if slice + BATCH_SIZE >= users_np.shape[0]:
                        break
                    static_loss = exe.run(
                        main,
                        feed={
                            users.name: users_np[slice:slice + BATCH_SIZE],
                            items.name: items_np[slice:slice + BATCH_SIZE],
                            labels.name: labels_np[slice:slice + BATCH_SIZE]
                        },
                        fetch_list=[loss])[0]
                    sys.stderr.write('static loss %s\n' % static_loss)
        """

        with fluid.imperative.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            deepcf = DeepCF('deepcf', num_users, num_items, matrix)
            sys.stderr.write('matrix: %s\n' % deepcf._rating_matrix._numpy())
            for e in range(NUM_EPOCHES):
                sys.stderr.write('epoch %d\n' % e)
                for slice in range(0, BATCH_SIZE * NUM_BATCHES, BATCH_SIZE):
                    prediction = deepcf(
                        to_variable(users_np[slice:slice + BATCH_SIZE]),
                        to_variable(items_np[slice:slice + BATCH_SIZE]))
                    loss = fluid.layers.reduce_sum(
                        fluid.layers.log_loss(prediction,
                                              to_variable(labels_np[
                                                  slice:slice + BATCH_SIZE])))
                    loss._backward()
                    adam = fluid.optimizer.AdamOptimizer(0.01)
                    adam.minimize(loss)
                    deepcf.clear_gradients()
                    dy_loss = loss._numpy()
                    sys.stderr.write('dynamic loss: %s\n' % dy_loss)
            sys.stderr.write('matrix: %s\n' % deepcf._rating_matrix._numpy())

        self.assertEqual(static_loss, dy_loss)


if __name__ == '__main__':
    unittest.main()
