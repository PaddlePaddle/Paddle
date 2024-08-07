# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
from paddle.distributed.fleet.meta_optimizers.meta_optimizer_base import (
    MetaOptimizerBase,
)

paddle.enable_static()


class TestFleetMetaOptimizerBase(unittest.TestCase):
    def net(main_prog, startup_prog):
        with base.program_guard(main_prog, startup_prog):
            with base.unique_name.guard():
                role = role_maker.PaddleCloudRoleMaker(is_collective=True)
                fleet.init(role)
                input_x = paddle.static.data(
                    name="x", shape=[-1, 32], dtype='float32'
                )
                input_y = paddle.static.data(
                    name="y", shape=[-1, 1], dtype='int64'
                )

                fc_1 = paddle.static.nn.fc(
                    x=input_x, size=64, activation='tanh'
                )
                fc_2 = paddle.static.nn.fc(x=fc_1, size=256, activation='tanh')
                prediction = paddle.static.nn.fc(
                    x=[fc_2], size=2, activation='softmax'
                )
                cost = paddle.nn.functional.cross_entropy(
                    input=prediction,
                    label=input_y,
                    reduction='none',
                    use_softmax=False,
                )
                avg_cost = paddle.mean(x=cost)

                optimizer = paddle.optimizer.SGD(learning_rate=0.01)
                opt = MetaOptimizerBase(optimizer)
                opt_ops, params_grads = opt.minimize(avg_cost)
                opt.apply_optimize(
                    avg_cost,
                    paddle.static.default_startup_program(),
                    params_grads,
                )

    net(base.default_startup_program(), base.default_main_program())


if __name__ == "__main__":
    unittest.main()
