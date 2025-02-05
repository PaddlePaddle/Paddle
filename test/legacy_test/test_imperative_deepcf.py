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

import os
import random
import sys
import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base
from paddle.base import core
from paddle.nn import Linear


class DMF(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._user_latent = Linear(1000, 256)
        self._item_latent = Linear(100, 256)

        self._user_layers = []
        self._item_layers = []
        self._hid_sizes = [128, 64]
        for i in range(len(self._hid_sizes)):
            self._user_layers.append(
                self.add_sublayer(
                    f'user_layer_{i}',
                    Linear(
                        256 if i == 0 else self._hid_sizes[i - 1],
                        self._hid_sizes[i],
                    ),
                )
            )
            self._user_layers.append(
                self.add_sublayer(
                    f'user_layer_act_{i}',
                    paddle.nn.ReLU(),
                )
            )
            self._item_layers.append(
                self.add_sublayer(
                    f'item_layer_{i}',
                    Linear(
                        256 if i == 0 else self._hid_sizes[i - 1],
                        self._hid_sizes[i],
                    ),
                )
            )
            self._item_layers.append(
                self.add_sublayer(
                    f'item_layer_act_{i}',
                    paddle.nn.ReLU(),
                )
            )

    def forward(self, users, items):
        users = self._user_latent(users)
        items = self._item_latent(items)

        for ul, il in zip(self._user_layers, self._item_layers):
            users = ul(users)
            items = il(items)
        return paddle.multiply(users, items)


class MLP(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._user_latent = Linear(1000, 256)
        self._item_latent = Linear(100, 256)
        self._match_layers = []
        self._hid_sizes = [128, 64]
        for i in range(len(self._hid_sizes)):
            self._match_layers.append(
                self.add_sublayer(
                    f'match_layer_{i}',
                    Linear(
                        256 * 2 if i == 0 else self._hid_sizes[i - 1],
                        self._hid_sizes[i],
                    ),
                )
            )
            self._match_layers.append(
                self.add_sublayer(
                    f'match_layer_act_{i}',
                    paddle.nn.ReLU(),
                )
            )

    def forward(self, users, items):
        users = self._user_latent(users)
        items = self._item_latent(items)
        match_vec = paddle.concat([users, items], axis=len(users.shape) - 1)
        for l in self._match_layers:
            match_vec = l(match_vec)
        return match_vec


class DeepCF(paddle.nn.Layer):
    def __init__(self, num_users, num_items, matrix):
        super().__init__()
        self._num_users = num_users
        self._num_items = num_items
        self._rating_matrix = self.create_parameter(
            attr=base.ParamAttr(trainable=False),
            shape=matrix.shape,
            dtype=matrix.dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Assign(matrix),
        )
        self._rating_matrix.stop_gradient = True

        self._mlp = MLP()
        self._dmf = DMF()
        self._match_fc = Linear(128, 1)

    def forward(self, users, items):
        # users_emb = self._user_emb(users)
        # items_emb = self._item_emb(items)

        users_emb = paddle.gather(self._rating_matrix, users)
        items_emb = paddle.gather(
            paddle.transpose(self._rating_matrix, [1, 0]), items
        )
        users_emb.stop_gradient = True
        items_emb.stop_gradient = True

        mlp_predictive = self._mlp(users_emb, items_emb)
        dmf_predictive = self._dmf(users_emb, items_emb)
        predictive = paddle.concat(
            [mlp_predictive, dmf_predictive], axis=len(mlp_predictive.shape) - 1
        )
        prediction = self._match_fc(predictive)
        prediction = paddle.nn.functional.sigmoid(prediction)
        return prediction


class TestDygraphDeepCF(unittest.TestCase):
    def setUp(self):
        # Can use Amusic dataset as the DeepCF describes.
        self.data_path = os.environ.get('DATA_PATH', '')

        self.batch_size = int(os.environ.get('BATCH_SIZE', 128))
        self.num_batches = int(os.environ.get('NUM_BATCHES', 5))
        self.num_epochs = int(os.environ.get('NUM_EPOCHS', 1))

    def get_data(self):
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
        return (
            np.expand_dims(users_np, -1),
            np.expand_dims(items_np, -1),
            np.expand_dims(labels_np, -1),
            NUM_USERS,
            NUM_ITEMS,
            matrix,
        )

    def load_data(self):
        sys.stderr.write(f'loading from {self.data_path}\n')
        likes = {}
        num_users = -1
        num_items = -1
        with open(self.data_path, 'r') as f:
            for l in f:
                uid, iid, rating = (int(v) for v in l.split('\t'))
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
        return (
            np.expand_dims(users_np, -1),
            np.expand_dims(items_np, -1),
            np.expand_dims(labels_np, -1),
            num_users,
            num_items,
            matrix,
        )

    def test_deefcf(self):
        seed = 90
        if self.data_path:
            (
                users_np,
                items_np,
                labels_np,
                num_users,
                num_items,
                matrix,
            ) = self.load_data()
        else:
            (
                users_np,
                items_np,
                labels_np,
                num_users,
                num_items,
                matrix,
            ) = self.get_data()
        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)
        startup = base.Program()
        main = base.Program()

        scope = base.core.Scope()
        with new_program_scope(main=main, startup=startup, scope=scope):
            users = paddle.static.data('users', [-1, 1], dtype='int32')
            items = paddle.static.data('items', [-1, 1], dtype='int32')
            labels = paddle.static.data('labels', [-1, 1], dtype='float32')

            deepcf = DeepCF(num_users, num_items, matrix)
            prediction = deepcf(users, items)
            loss = paddle.sum(paddle.nn.functional.log_loss(prediction, labels))
            adam = paddle.optimizer.Adam(0.01)
            adam.minimize(loss)

            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            exe.run(startup)
            for e in range(self.num_epochs):
                sys.stderr.write(f'epoch {e}\n')
                for slice in range(
                    0, self.batch_size * self.num_batches, self.batch_size
                ):
                    if slice + self.batch_size >= users_np.shape[0]:
                        break
                    static_loss = exe.run(
                        main,
                        feed={
                            users.name: users_np[
                                slice : slice + self.batch_size
                            ],
                            items.name: items_np[
                                slice : slice + self.batch_size
                            ],
                            labels.name: labels_np[
                                slice : slice + self.batch_size
                            ],
                        },
                        fetch_list=[loss],
                    )[0]
                    sys.stderr.write(f'static loss {static_loss}\n')

        with base.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            deepcf = DeepCF(num_users, num_items, matrix)
            adam = paddle.optimizer.Adam(0.01, parameters=deepcf.parameters())
            for e in range(self.num_epochs):
                sys.stderr.write(f'epoch {e}\n')
                for slice in range(
                    0, self.batch_size * self.num_batches, self.batch_size
                ):
                    if slice + self.batch_size >= users_np.shape[0]:
                        break
                    prediction = deepcf(
                        paddle.to_tensor(
                            users_np[slice : slice + self.batch_size]
                        ),
                        paddle.to_tensor(
                            items_np[slice : slice + self.batch_size]
                        ),
                    )
                    loss = paddle.sum(
                        paddle.nn.functional.log_loss(
                            prediction,
                            paddle.to_tensor(
                                labels_np[slice : slice + self.batch_size]
                            ),
                        )
                    )
                    loss.backward()
                    adam.minimize(loss)
                    deepcf.clear_gradients()
                    dy_loss = loss.numpy()
                    sys.stderr.write(f'dynamic loss: {slice} {dy_loss}\n')

        with base.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            deepcf2 = DeepCF(num_users, num_items, matrix)
            adam2 = paddle.optimizer.Adam(0.01, parameters=deepcf2.parameters())
            base.set_flags({'FLAGS_sort_sum_gradient': True})
            for e in range(self.num_epochs):
                sys.stderr.write(f'epoch {e}\n')
                for slice in range(
                    0, self.batch_size * self.num_batches, self.batch_size
                ):
                    if slice + self.batch_size >= users_np.shape[0]:
                        break
                    prediction2 = deepcf2(
                        paddle.to_tensor(
                            users_np[slice : slice + self.batch_size]
                        ),
                        paddle.to_tensor(
                            items_np[slice : slice + self.batch_size]
                        ),
                    )
                    loss2 = paddle.sum(
                        paddle.nn.functional.log_loss(
                            prediction2,
                            paddle.to_tensor(
                                labels_np[slice : slice + self.batch_size]
                            ),
                        )
                    )
                    loss2.backward()
                    adam2.minimize(loss2)
                    deepcf2.clear_gradients()
                    dy_loss2 = loss2.numpy()
                    sys.stderr.write(f'dynamic loss: {slice} {dy_loss2}\n')

        with base.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            deepcf = DeepCF(num_users, num_items, matrix)
            adam = paddle.optimizer.Adam(0.01, parameters=deepcf.parameters())

            for e in range(self.num_epochs):
                sys.stderr.write(f'epoch {e}\n')
                for slice in range(
                    0, self.batch_size * self.num_batches, self.batch_size
                ):
                    if slice + self.batch_size >= users_np.shape[0]:
                        break
                    prediction = deepcf(
                        paddle.to_tensor(
                            users_np[slice : slice + self.batch_size]
                        ),
                        paddle.to_tensor(
                            items_np[slice : slice + self.batch_size]
                        ),
                    )
                    loss = paddle.sum(
                        paddle.nn.functional.log_loss(
                            prediction,
                            paddle.to_tensor(
                                labels_np[slice : slice + self.batch_size]
                            ),
                        )
                    )
                    loss.backward()
                    adam.minimize(loss)
                    deepcf.clear_gradients()
                    eager_loss = loss.numpy()
                    sys.stderr.write(f'eager loss: {slice} {eager_loss}\n')

        self.assertEqual(static_loss, dy_loss)
        self.assertEqual(static_loss, dy_loss2)
        self.assertEqual(static_loss, eager_loss)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
