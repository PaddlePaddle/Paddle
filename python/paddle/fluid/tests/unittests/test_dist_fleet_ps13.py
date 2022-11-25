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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
import os

os.environ["WITH_DISTRIBUTE"] = "ON"

import unittest
<<<<<<< HEAD
=======
import tempfile
import shutil
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

import paddle
import paddle.fluid as fluid
import paddle.distributed.fleet.base.role_maker as role_maker
import paddle.distributed.fleet as fleet

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


# this unittest is tested for SparseSharedAdamSGDRule
class TestPSPassWithBow(unittest.TestCase):
<<<<<<< HEAD
    def net(self):
=======

    def net(self):

>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        def get_acc(cos_q_nt, cos_q_pt, batch_size):
            cond = fluid.layers.less_than(cos_q_nt, cos_q_pt)
            cond = fluid.layers.cast(cond, dtype='float64')
            cond_3 = fluid.layers.reduce_sum(cond)
<<<<<<< HEAD
            acc = fluid.layers.elementwise_div(
                cond_3,
                fluid.layers.fill_constant(
                    shape=[1], value=batch_size * 1.0, dtype='float64'
                ),
                name="simnet_acc",
            )
=======
            acc = fluid.layers.elementwise_div(cond_3,
                                               fluid.layers.fill_constant(
                                                   shape=[1],
                                                   value=batch_size * 1.0,
                                                   dtype='float64'),
                                               name="simnet_acc")
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
            return acc

        def get_loss(cos_q_pt, cos_q_nt):
            loss_op1 = fluid.layers.elementwise_sub(
<<<<<<< HEAD
                fluid.layers.fill_constant_batch_size_like(
                    input=cos_q_pt, shape=[-1, 1], value=margin, dtype='float32'
                ),
                cos_q_pt,
            )
            loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
            loss_op3 = paddle.maximum(
                fluid.layers.fill_constant_batch_size_like(
                    input=loss_op2, shape=[-1, 1], value=0.0, dtype='float32'
                ),
                loss_op2,
            )
=======
                fluid.layers.fill_constant_batch_size_like(input=cos_q_pt,
                                                           shape=[-1, 1],
                                                           value=margin,
                                                           dtype='float32'),
                cos_q_pt)
            loss_op2 = fluid.layers.elementwise_add(loss_op1, cos_q_nt)
            loss_op3 = fluid.layers.elementwise_max(
                fluid.layers.fill_constant_batch_size_like(input=loss_op2,
                                                           shape=[-1, 1],
                                                           value=0.0,
                                                           dtype='float32'),
                loss_op2)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
            avg_cost = fluid.layers.mean(loss_op3)
            return avg_cost

        is_distributed = False
        is_sparse = True

        # query
<<<<<<< HEAD
        q = fluid.layers.data(
            name="query_ids", shape=[1], dtype="int64", lod_level=1
        )
=======
        q = fluid.layers.data(name="query_ids",
                              shape=[1],
                              dtype="int64",
                              lod_level=1)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        # embedding
        q_emb = fluid.contrib.layers.sparse_embedding(
            input=q,
            size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__emb__",
<<<<<<< HEAD
                learning_rate=emb_lr,
            ),
        )
        q_emb = paddle.reshape(q_emb, [-1, emb_dim])
        # vsum
        q_sum = fluid.layers.sequence_pool(input=q_emb, pool_type='sum')
        q_ss = paddle.nn.functional.softsign(q_sum)
=======
                learning_rate=emb_lr))
        q_emb = fluid.layers.reshape(q_emb, [-1, emb_dim])
        # vsum
        q_sum = fluid.layers.sequence_pool(input=q_emb, pool_type='sum')
        q_ss = fluid.layers.softsign(q_sum)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        # fc layer after conv
        q_fc = fluid.layers.fc(
            input=q_ss,
            size=hid_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__q_fc__",
<<<<<<< HEAD
                learning_rate=base_lr,
            ),
        )
        # label data
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        # pt
        pt = fluid.layers.data(
            name="pos_title_ids", shape=[1], dtype="int64", lod_level=1
        )
=======
                learning_rate=base_lr))
        # label data
        label = fluid.layers.data(name="label", shape=[1], dtype="int64")
        # pt
        pt = fluid.layers.data(name="pos_title_ids",
                               shape=[1],
                               dtype="int64",
                               lod_level=1)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        # embedding
        pt_emb = fluid.contrib.layers.sparse_embedding(
            input=pt,
            size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__emb__",
<<<<<<< HEAD
                learning_rate=emb_lr,
            ),
        )
        pt_emb = paddle.reshape(pt_emb, [-1, emb_dim])
        # vsum
        pt_sum = fluid.layers.sequence_pool(input=pt_emb, pool_type='sum')
        pt_ss = paddle.nn.functional.softsign(pt_sum)
=======
                learning_rate=emb_lr))
        pt_emb = fluid.layers.reshape(pt_emb, [-1, emb_dim])
        # vsum
        pt_sum = fluid.layers.sequence_pool(input=pt_emb, pool_type='sum')
        pt_ss = fluid.layers.softsign(pt_sum)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        # fc layer
        pt_fc = fluid.layers.fc(
            input=pt_ss,
            size=hid_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__fc__",
<<<<<<< HEAD
                learning_rate=base_lr,
            ),
            bias_attr=fluid.ParamAttr(name="__fc_b__"),
        )
        # nt
        nt = fluid.layers.data(
            name="neg_title_ids", shape=[1], dtype="int64", lod_level=1
        )
=======
                learning_rate=base_lr),
            bias_attr=fluid.ParamAttr(name="__fc_b__"))
        # nt
        nt = fluid.layers.data(name="neg_title_ids",
                               shape=[1],
                               dtype="int64",
                               lod_level=1)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        # embedding
        nt_emb = fluid.contrib.layers.sparse_embedding(
            input=nt,
            size=[dict_dim, emb_dim],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__emb__",
<<<<<<< HEAD
                learning_rate=emb_lr,
            ),
        )
        nt_emb = paddle.reshape(nt_emb, [-1, emb_dim])
        # vsum
        nt_sum = fluid.layers.sequence_pool(input=nt_emb, pool_type='sum')
        nt_ss = paddle.nn.functional.softsign(nt_sum)
=======
                learning_rate=emb_lr))
        nt_emb = fluid.layers.reshape(nt_emb, [-1, emb_dim])
        # vsum
        nt_sum = fluid.layers.sequence_pool(input=nt_emb, pool_type='sum')
        nt_ss = fluid.layers.softsign(nt_sum)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        # fc layer
        nt_fc = fluid.layers.fc(
            input=nt_ss,
            size=hid_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01),
                name="__fc__",
<<<<<<< HEAD
                learning_rate=base_lr,
            ),
            bias_attr=fluid.ParamAttr(name="__fc_b__"),
        )
=======
                learning_rate=base_lr),
            bias_attr=fluid.ParamAttr(name="__fc_b__"))
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        cos_q_pt = fluid.layers.cos_sim(q_fc, pt_fc)
        cos_q_nt = fluid.layers.cos_sim(q_fc, nt_fc)
        # loss
        avg_cost = get_loss(cos_q_pt, cos_q_nt)
        # acc
        acc = get_acc(cos_q_nt, cos_q_pt, batch_size)
        return [avg_cost, acc, cos_q_pt]

    def test(self):
        os.environ["PADDLE_PSERVER_NUMS"] = "2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
<<<<<<< HEAD
            "PADDLE_PSERVERS_IP_PORT_LIST"
        ] = "127.0.0.1:36001,127.0.0.2:36001"
=======
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001,127.0.0.2:36001"
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        os.environ["TRAINING_ROLE"] = "PSERVER"

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        loss, acc, _ = self.net()

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True

        configs = {}
        configs['__emb__'] = {
<<<<<<< HEAD
            "table_parameters.__emb__.accessor.embed_sgd_param.name": "SparseSharedAdamSGDRule",
            "table_parameters.__emb__.accessor.embedx_sgd_param.name": "SparseSharedAdamSGDRule",
=======
            "table_parameters.__emb__.accessor.embed_sgd_param.name":
            "SparseSharedAdamSGDRule",
            "table_parameters.__emb__.accessor.embedx_sgd_param.name":
            "SparseSharedAdamSGDRule",
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        }
        strategy.sparse_table_configs = configs
        optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(loss)

        fleet.init_server()


if __name__ == '__main__':
    unittest.main()
