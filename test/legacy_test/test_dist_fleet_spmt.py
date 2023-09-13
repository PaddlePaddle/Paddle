#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import paddle
from paddle import base

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


class TestSPMT(unittest.TestCase):
    def net(self):
        def get_acc(cos_q_nt, cos_q_pt, batch_size):
            cond = paddle.less_than(cos_q_nt, cos_q_pt)
            cond = paddle.cast(cond, dtype='float64')
            cond_3 = paddle.sum(cond)
            acc = paddle.divide(
                cond_3,
                paddle.tensor.fill_constant(
                    shape=[1], value=batch_size * 1.0, dtype='float64'
                ),
                name="simnet_acc",
            )
            return acc

        def get_loss(cos_q_pt, cos_q_nt):
            fill_shape = [-1, 1]
            fill_shape[0] = paddle.shape(cos_q_pt)[0].item()
            loss_op1 = paddle.subtract(
                paddle.full(
                    shape=fill_shape, fill_value=margin, dtype='float32'
                ),
                cos_q_pt,
            )
            loss_op2 = paddle.add(loss_op1, cos_q_nt)
            fill_shape = [-1, 1]
            fill_shape[0] = paddle.shape(loss_op2)[0].item()
            loss_op3 = paddle.maximum(
                paddle.full(shape=fill_shape, fill_value=0.0, dtype='float32'),
                loss_op2,
            )
            avg_cost = paddle.mean(loss_op3)
            return avg_cost

        is_distributed = False
        is_sparse = True

        # query
        q = paddle.static.data(
            name="1", shape=[-1, 1], dtype="int64", lod_level=1
        )
        # embedding
        q_emb = paddle.static.nn.sparse_embedding(
            input=q,
            size=[dict_dim, emb_dim],
            param_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01),
                name="__emb__",
                learning_rate=emb_lr,
            ),
        )
        q_emb = paddle.reshape(q_emb, [-1, emb_dim])
        # vsum
        q_sum = paddle.static.nn.sequence_lod.sequence_pool(
            input=q_emb, pool_type='sum'
        )
        q_ss = paddle.nn.functional.softsign(q_sum)
        # fc layer after conv
        q_fc = paddle.static.nn.fc(
            x=q_ss,
            size=hid_dim,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01),
                name="__q_fc__",
                learning_rate=base_lr,
            ),
        )
        # label data
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")
        # pt
        pt = paddle.static.data(
            name="2", shape=[-1, 1], dtype="int64", lod_level=1
        )
        # embedding
        pt_emb = paddle.static.nn.sparse_embedding(
            input=pt,
            size=[dict_dim, emb_dim],
            param_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01),
                name="__emb__",
                learning_rate=emb_lr,
            ),
        )
        pt_emb = paddle.reshape(pt_emb, [-1, emb_dim])
        # vsum
        pt_sum = paddle.static.nn.sequence_lod.sequence_pool(
            input=pt_emb, pool_type='sum'
        )
        pt_ss = paddle.nn.functional.softsign(pt_sum)
        # fc layer
        pt_fc = paddle.static.nn.fc(
            x=pt_ss,
            size=hid_dim,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01),
                name="__fc__",
                learning_rate=base_lr,
            ),
            bias_attr=base.ParamAttr(name="__fc_b__"),
        )
        # nt
        nt = paddle.static.data(
            name="3", shape=[-1, 1], dtype="int64", lod_level=1
        )
        # embedding
        nt_emb = paddle.static.nn.sparse_embedding(
            input=nt,
            size=[dict_dim, emb_dim],
            param_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01),
                name="__emb__",
                learning_rate=emb_lr,
            ),
        )
        nt_emb = paddle.reshape(nt_emb, [-1, emb_dim])
        # vsum
        nt_sum = paddle.static.nn.sequence_lod.sequence_pool(
            input=nt_emb, pool_type='sum'
        )
        nt_ss = paddle.nn.functional.softsign(nt_sum)
        # fc layer
        nt_fc = paddle.static.nn.fc(
            x=nt_ss,
            size=hid_dim,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.01),
                name="__fc__",
                learning_rate=base_lr,
            ),
            bias_attr=base.ParamAttr(name="__fc_b__"),
        )
        cos_q_pt = paddle.nn.functional.cosine_similarity(q_fc, pt_fc)
        cos_q_nt = paddle.nn.functional.cosine_similarity(q_fc, nt_fc)
        # loss
        avg_cost = get_loss(cos_q_pt, cos_q_nt)
        # acc
        acc = get_acc(cos_q_nt, cos_q_pt, batch_size)
        return [avg_cost, acc, cos_q_pt]

    # def test(self):
    #    os.environ["PADDLE_PSERVER_NUMS"] = "2"
    #    os.environ["PADDLE_TRAINERS_NUM"] = "2"
    #    os.environ["POD_IP"] = "127.0.0.1"
    #    os.environ["PADDLE_PORT"] = "36001"
    #    os.environ["PADDLE_TRAINER_ID"] = "0"
    #    os.environ["PADDLE_TRAINERS_NUM"] = "2"
    #    os.environ[
    #        "PADDLE_TRAINER_ENDPOINTS"
    #    ] = "127.0.0.1:36001,127.0.0.2:36001"
    #    os.environ[
    #        "PADDLE_PSERVERS_IP_PORT_LIST"
    #    ] = "127.0.0.1:36002,127.0.0.2:36002"
    #    os.environ["TRAINING_ROLE"] = "TRAINER"
    #    os.environ["FLAGS_selected_gpus"] = "0"
    #    role = role_maker.PaddleCloudRoleMaker()
    #    fleet.init(role)
    #    loss, acc, _ = self.net()
    #
    #    strategy = paddle.distributed.fleet.DistributedStrategy()
    #    configs = {"use_ps_gpu": 1, "launch_barrier": False}
    #    strategy.a_sync_configs = configs
    #    strategy.a_sync = True
    #    optimizer = paddle.optimizer.Adam(learning_rate=0.01)
    #    optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
    #    optimizer.minimize(loss)

    def get_dist_env(self):
        trainer_id = int(os.getenv('PADDLE_TRAINER_ID', '0'))
        trainer_endpoints = ''
        current_endpoint = ''
        num_trainers = 0
        if os.getenv('PADDLE_TRAINER_ENDPOINTS'):
            trainer_endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS')
            current_endpoint = trainer_endpoints.split(',')[trainer_id]
            num_trainers = len(trainer_endpoints.split(','))

        return {
            'trainer_id': trainer_id,
            'num_trainers': num_trainers,
            'current_endpoint': current_endpoint,
            'trainer_endpoints': trainer_endpoints,
        }

    def test_SingleProcessMultiThread(self):
        """
        Testcase for SingleProcessMultiThread
        """
        os.environ["PADDLE_PSERVER_NUMS"] = "2"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"
        ] = "127.0.0.1:36001,127.0.0.2:36001"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"
        ] = "127.0.0.1:36002,127.0.0.2:36002"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["FLAGS_selected_gpus"] = "0"
        os.environ["PADDLE_FUSE_ALLREDUCE"] = "1"
        os.environ["PADDLE_LOSS_SCALE"] = "1"

        startup_program = base.Program()
        main_program = base.Program()
        with base.program_guard(main_program, startup_program):
            with base.unique_name.guard():
                loss, acc, _ = self.net()
        optimizer = paddle.optimizer.Adam(learning_rate=0.01)
        optimizer.minimize(loss)
        print("===main_program====")
        print(main_program)
        print("===main_program====")
        from paddle.distributed.transpiler.collective import (
            SingleProcessMultiThread,
        )

        t = SingleProcessMultiThread()
        env = self.get_dist_env()
        t.transpile(
            startup_program=startup_program,
            main_program=main_program,
            rank=env["trainer_id"],
            endpoints=env["trainer_endpoints"],
            current_endpoint=env['current_endpoint'],
            wait_port=False,
        )
        param_cnt = t._get_update_param_count()
        print("param_cnt:", param_cnt)


if __name__ == '__main__':
    unittest.main()
