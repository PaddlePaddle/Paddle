#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import tempfile
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.base import role_maker


class SparseLoadOp(unittest.TestCase):
    """Test load operator."""

    def net(self, emb_array, fc_array):
        with base.unique_name.guard():
            dense_input = paddle.static.data(
                'input', shape=[None, 1], dtype="int64"
            )

            emb = paddle.nn.Embedding(
                num_embeddings=10,
                embedding_dim=10,
                sparse=True,
                weight_attr=base.ParamAttr(
                    name="embedding",
                    initializer=paddle.nn.initializer.Assign(emb_array),
                ),
            )(dense_input)
            linear = paddle.nn.Linear(
                in_features=emb.shape[-1],
                out_features=10,
                weight_attr=base.ParamAttr(
                    name='fc',
                    initializer=paddle.nn.initializer.Assign(fc_array),
                ),
            )(emb)
            fc1 = paddle.nn.ReLU()(linear)
            loss = paddle.mean(fc1)
        return loss

    def save_origin_model(self, emb_array, fc_array):
        startup_program = base.framework.Program()
        test_program = base.framework.Program()
        with base.framework.program_guard(test_program, startup_program):
            with base.unique_name.guard():
                loss = self.net(emb_array, fc_array)
                optimizer = paddle.optimizer.Adam(1e-3)
                optimizer.minimize(loss)

                exe = base.Executor(base.CPUPlace())
                exe.run(startup_program)
                model_path = tempfile.mkdtemp()
                paddle.distributed.io.save_persistables(
                    executor=exe, dirname=model_path
                )
        return model_path


@unittest.skip(reason="Skip unstable ut, need rewrite with new implement")
class TestSparseLoadOpCase1(SparseLoadOp):
    def test_2ps_0_load(self):
        # init No.0 server env
        env = {}
        env["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        env["PADDLE_TRAINERS_NUM"] = str(2)
        env["TRAINING_ROLE"] = "PSERVER"
        env["PADDLE_PORT"] = "4001"
        env["POD_IP"] = "127.0.0.1"
        for k, v in env.items():
            os.environ[k] = str(v)
        """
        array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
        """
        emb_array = np.arange(0, 1, 0.1).repeat(10).reshape(10, 10)
        fc_array = np.arange(0, 1, 0.1).repeat(10).reshape(10, 10)
        model_path = self.save_origin_model(emb_array, fc_array)

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        loss = self.net(emb_array, fc_array)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        optimizer = paddle.optimizer.Adam(1e-3)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)
        fleet.init_server(model_path)

        fc_w = np.array(base.global_scope().find_var("fc").get_tensor())

        emb = np.array(
            base.global_scope().find_var("embedding.block0").get_tensor()
        )

        assert fc_w.all() == fc_array.all()
        assert emb.all() == emb_array[::2].all()
        shutil.rmtree(model_path)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
