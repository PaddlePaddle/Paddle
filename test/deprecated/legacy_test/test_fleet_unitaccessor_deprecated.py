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
"""Test fleet."""

import os
import unittest

import paddle

paddle.enable_static()


class TestFleet1(unittest.TestCase):
    """
    Test cases for fleet minimize.
    """

    def setUp(self):
        """Set up, set envs."""
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = (
            "127.0.0.1:36001,127.0.0.2:36001"
        )

    def test_pslib_1(self):
        """Test cases for pslib."""
        from paddle import base
        from paddle.incubate.distributed.fleet.parameter_server.pslib import (
            fleet,
        )
        from paddle.incubate.distributed.fleet.role_maker import (
            GeneralRoleMaker,
        )

        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        role_maker = GeneralRoleMaker()
        # role_maker.generate_role()
        place = base.CPUPlace()
        exe = base.Executor(place)
        # fleet.init(role_maker)
        train_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(train_program, startup_program):
            show = paddle.static.data(name="show", shape=[-1, 1], dtype="int64")
            emb = paddle.static.nn.embedding(
                input=show,
                size=[1, 1],
                is_sparse=True,
                is_distributed=True,
                param_attr=base.ParamAttr(name="embedding"),
            )
            fc = paddle.static.nn.fc(x=emb, size=1, activation=None)
            label = paddle.static.data(
                name="click", shape=[-1, 1], dtype="int64"
            )
            label_cast = paddle.cast(label, dtype='float32')
            cost = paddle.nn.functional.log_loss(fc, label_cast)

        strategy = {}
        strategy["embedding"] = {}
        strategy["embedding"]["sparse_accessor_class"] = "DownpourUnitAccessor"
        strategy["embedding"]["embed_sparse_optimizer"] = "naive"
        try:
            adam1 = paddle.optimizer.Adam(learning_rate=0.000005)
            adam1 = fleet.distributed_optimizer(adam1, strategy=strategy)
            adam1.minimize([cost], [scope])

            strategy["embedding"]["embed_sparse_optimizer"] = "adagrad"
            adam2 = paddle.optimizer.Adam(learning_rate=0.000005)
            adam2 = fleet.distributed_optimizer(adam2, strategy=strategy)
            adam2.minimize([cost], [scope])

            strategy["embedding"]["embed_sparse_optimizer"] = "adam"
            adam3 = paddle.optimizer.Adam(learning_rate=0.000005)
            adam3 = fleet.distributed_optimizer(adam3, strategy=strategy)
            adam3.minimize([cost], [scope])
        except:
            print("do not support pslib test, skip")
            return


if __name__ == "__main__":
    unittest.main()
