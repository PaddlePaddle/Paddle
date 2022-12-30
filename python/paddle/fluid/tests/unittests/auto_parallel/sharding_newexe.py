# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import random
import unittest

import numpy as np
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed.fleet import auto
from paddle.fluid.dygraph.parallel import ParallelEnv

paddle.enable_static()


def apply_pass(use_sharding=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    if use_sharding:
        sharding = strategy.sharding
        sharding.enable = True
        sharding.degree = 2
        sharding.stage = 2
        sharding.enable_overlap = True
        sharding.param_comm_stream_num = 2
        sharding.grad_comm_stream_num = 2
        sharding.param_bucket_size_numel = 512 * 512
        sharding.grad_bucket_size_numel = 128 * 128
        sharding.partition_algor = 'use_order'

    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestShardingStage2WithNewEXE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.batch_num = 5
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_sharding=False):
        reset_prog()

        strategy = apply_pass(use_sharding)
        clip = paddle.nn.ClipGradByNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("dp")
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_result(self, dp_params, sharding_params):
        assert len(dp_params) == len(sharding_params)
        for dp_p, sharding_p in zip(dp_params, sharding_params):
            np.testing.assert_allclose(
                dp_p,
                sharding_p,
                rtol=1e-05,
                atol=1e-08,
                err_msg='gradient clip by global norm has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                    dp_p, sharding_p, dp_p - sharding_p
                ),
            )

    def test_grad_clip(self):
        # dp2 training
        dp_engine = self.get_engine(False)
        dp_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        with open(
            "./main_prog.txt.{}".format(paddle.distributed.get_rank()), "w+"
        ) as f:
            f.write(str(dp_engine.main_program))

        # dp2sharding2 training
        sharding_engine = self.get_engine(True)
        sharding_engine.fit(
            self.dataset,
            3,
            epochs=1,
            steps_per_epoch=self.batch_num,
            log_freq=1,
            batch_size=self.batch_size,
        )
        with open(
            "./sharding_prog.txt.{}".format(paddle.distributed.get_rank()), "w+"
        ) as f:
            f.write(str(sharding_engine.main_program))

        # self.check_result(dp_param_values, sharding_param_values)


if __name__ == "__main__":
    unittest.main()
