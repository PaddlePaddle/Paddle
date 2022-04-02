#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
os.environ["WITH_DISTRIBUTE"] = "ON"
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet
import unittest
import paddle

paddle.enable_static()

# For Net
base_lr = 0.2
emb_lr = base_lr * 3
dict_dim = 1500
emb_dim = 128
hid_dim = 128
margin = 0.1
sample_rate = 1
batch_size = 4


class TestExponentialDecay(unittest.TestCase):
    def net(self):
        input_data = paddle.static.data(
            name="sparse_input", shape=[None, 1], dtype="int64")
        input_label = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")
        label = paddle.cast(input_label, dtype="float32")
        embedding = paddle.static.nn.embedding(
            input_data, is_sparse=True, size=[1000, 128])

        fc1 = paddle.static.nn.fc(embedding, size=1024, activation="relu")
        fc2 = paddle.static.nn.fc(fc1, size=512, activation="relu")
        fc3 = paddle.static.nn.fc(fc2, size=256, activation="relu")
        predict = paddle.static.nn.fc(fc3, size=2, activation="softmax")
        label = paddle.cast(label, dtype="int64")
        cost = paddle.nn.functional.cross_entropy(input=predict, label=label)
        paddle.static.Print(cost, message="heter_cost")
        return cost

    def test(self):
        endpoints = [
            "127.0.0.1:36004", "127.0.0.1:36005", "127.0.0.1:36006",
            "127.0.0.1:36007"
        ]

        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.SERVER,
            worker_num=2,
            server_endpoints=endpoints)

        fleet.init(role)
        loss = self.net()
        scheduler = paddle.optimizer.lr.InverseTimeDecay(
            learning_rate=base_lr, gamma=0.999, verbose=True)
        optimizer = fluid.optimizer.Adam(scheduler)

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize([loss])
        fleet.init_server()


if __name__ == '__main__':
    os.environ["GLOG_v"] = "4"
    os.environ["GLOG_logtostderr"] = "1"

    unittest.main()
