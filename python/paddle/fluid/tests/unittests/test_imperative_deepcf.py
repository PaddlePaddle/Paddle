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
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import Linear
from paddle.fluid.framework import _test_eager_guard

# Can use Amusic dataset as the DeepCF describes.
DATA_PATH = os.environ.get('DATA_PATH', '')

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 128))
NUM_BATCHES = int(os.environ.get('NUM_BATCHES', 5))
NUM_EPOCHES = int(os.environ.get('NUM_EPOCHES', 1))


class DMF(fluid.Layer):
    def __init__(self):
        super(DMF, self).__init__()
        self._user_latent = Linear(1000, 256)
        self._item_latent = Linear(100, 256)

        self._user_layers = []
        self._item_layers = []
        self._hid_sizes = [128, 64]
        for i in range(len(self._hid_sizes)):
            self._user_layers.append(
                self.add_sublayer(
                    'user_layer_%d' % i,
                    Linear(
                        256 if i == 0 else self._hid_sizes[i - 1],
                        self._hid_sizes[i],
                        act='relu')))
            self._item_layers.append(
                self.add_sublayer(
                    'item_layer_%d' % i,
                    Linear(
                        256 if i == 0 else self._hid_sizes[i - 1],
                        self._hid_sizes[i],
                        act='relu')))

    def forward(self, users, items):
        users = self._user_latent(users)
        items = self._item_latent(items)

        for ul, il in zip(self._user_layers, self._item_layers):
            users = ul(users)
            items = il(items)
        return fluid.layers.elementwise_mul(users, items)


class MLP(fluid.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self._user_latent = Linear(1000, 256)
        self._item_latent = Linear(100, 256)
        self._match_layers = []
        self._hid_sizes = [128, 64]
        for i in range(len(self._hid_sizes)):
            self._match_layers.append(
                self.add_sublayer(
                    'match_layer_%d' % i,
                    Linear(
                        256 * 2 if i == 0 else self._hid_sizes[i - 1],
                        self._hid_sizes[i],
                        act='relu')))

    def forward(self, users, items):
        users = self._user_latent(users)
        items = self._item_latent(items)
        match_vec = fluid.layers.concat(
            [users, items], axis=len(users.shape) - 1)
        for l in self._match_layers:
            match_vec = l(match_vec)
        return match_vec


class DeepCF(fluid.Layer):
    def __init__(self, num_users, num_items, matrix):
        super(DeepCF, self).__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._rating_matrix = self.create_parameter(
            attr=fluid.ParamAttr(trainable=False),
            shape=matrix.shape,
            dtype=matrix.dtype,
            is_bias=False,
            default_initializer=fluid.initializer.NumpyArrayInitializer(matrix))
        self._rating_matrix.stop_gradient = True

        self._mlp = MLP()
        self._dmf = DMF()
        self._match_fc = Linear(128, 1, act='sigmoid')

    def forward(self, users, items):
        # users_emb = self._user_emb(users)
        # items_emb = self._item_emb(items)
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
    NUM_USERS = 100
    NUM_ITEMS = 1000
    matrix = np.zeros([NUM_USERS, NUM_ITEMS], dtype=np.float32)

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


class TestDygraphDeepCF(unittest.TestCase):
    def test_deefcf(self):
        seed = 90
        if DATA_PATH:
            (users_np, items_np, labels_np, num_users, num_items,
             matrix) = load_data(DATA_PATH)
        else:
            (users_np, items_np, labels_np, num_users, num_items,
             matrix) = get_data()
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)
        startup = fluid.Program()
        main = fluid.Program()

        scope = fluid.core.Scope()
        with new_program_scope(main=main, startup=startup, scope=scope):
            users = fluid.layers.data('users', [1], dtype='int32')
            items = fluid.layers.data('items', [1], dtype='int32')
            labels = fluid.layers.data('labels', [1], dtype='float32')

            deepcf = DeepCF(num_users, num_items, matrix)
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

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            deepcf = DeepCF(num_users, num_items, matrix)
            adam = fluid.optimizer.AdamOptimizer(
                0.01, parameter_list=deepcf.parameters())
            for e in range(NUM_EPOCHES):
                sys.stderr.write('epoch %d\n' % e)
                for slice in range(0, BATCH_SIZE * NUM_BATCHES, BATCH_SIZE):
                    if slice + BATCH_SIZE >= users_np.shape[0]:
                        break
                    prediction = deepcf(
                        to_variable(users_np[slice:slice + BATCH_SIZE]),
                        to_variable(items_np[slice:slice + BATCH_SIZE]))
                    loss = fluid.layers.reduce_sum(
                        fluid.layers.log_loss(prediction,
                                              to_variable(labels_np[
                                                  slice:slice + BATCH_SIZE])))
                    loss.backward()
                    adam.minimize(loss)
                    deepcf.clear_gradients()
                    dy_loss = loss.numpy()
                    sys.stderr.write('dynamic loss: %s %s\n' % (slice, dy_loss))

        with fluid.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            deepcf2 = DeepCF(num_users, num_items, matrix)
            adam2 = fluid.optimizer.AdamOptimizer(
                0.01, parameter_list=deepcf2.parameters())
            fluid.set_flags({'FLAGS_sort_sum_gradient': True})
            for e in range(NUM_EPOCHES):
                sys.stderr.write('epoch %d\n' % e)
                for slice in range(0, BATCH_SIZE * NUM_BATCHES, BATCH_SIZE):
                    if slice + BATCH_SIZE >= users_np.shape[0]:
                        break
                    prediction2 = deepcf2(
                        to_variable(users_np[slice:slice + BATCH_SIZE]),
                        to_variable(items_np[slice:slice + BATCH_SIZE]))
                    loss2 = fluid.layers.reduce_sum(
                        fluid.layers.log_loss(prediction2,
                                              to_variable(labels_np[
                                                  slice:slice + BATCH_SIZE])))
                    loss2.backward()
                    adam2.minimize(loss2)
                    deepcf2.clear_gradients()
                    dy_loss2 = loss2.numpy()
                    sys.stderr.write('dynamic loss: %s %s\n' %
                                     (slice, dy_loss2))

        with fluid.dygraph.guard():
            with _test_eager_guard():
                paddle.seed(seed)
                paddle.framework.random._manual_program_seed(seed)
                fluid.default_startup_program().random_seed = seed
                fluid.default_main_program().random_seed = seed

                deepcf = DeepCF(num_users, num_items, matrix)
                adam = fluid.optimizer.AdamOptimizer(
                    0.01, parameter_list=deepcf.parameters())

                for e in range(NUM_EPOCHES):
                    sys.stderr.write('epoch %d\n' % e)
                    for slice in range(0, BATCH_SIZE * NUM_BATCHES, BATCH_SIZE):
                        if slice + BATCH_SIZE >= users_np.shape[0]:
                            break
                        prediction = deepcf(
                            to_variable(users_np[slice:slice + BATCH_SIZE]),
                            to_variable(items_np[slice:slice + BATCH_SIZE]))
                        loss = fluid.layers.reduce_sum(
                            fluid.layers.log_loss(prediction,
                                                  to_variable(
                                                      labels_np[slice:slice +
                                                                BATCH_SIZE])))
                        loss.backward()
                        adam.minimize(loss)
                        deepcf.clear_gradients()
                        eager_loss = loss.numpy()
                        sys.stderr.write('eager loss: %s %s\n' %
                                         (slice, eager_loss))

        self.assertEqual(static_loss, dy_loss)
        self.assertEqual(static_loss, dy_loss2)
        self.assertEqual(static_loss, eager_loss)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
