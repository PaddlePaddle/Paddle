# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass(enable=True, schedule_mode="FThenB"):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    pipeline = strategy.pipeline
    pipeline.enable = enable
    pipeline.schedule_mode = schedule_mode
    pipeline.accumulate_steps = 2

    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class Test1F1BPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 2
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, enable=True, schedule_mode="FThenB"):
        reset_prog()

        strategy = apply_pass(enable, schedule_mode)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("pp")

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=self.rtol,
            atol=self.atol,
            err_msg='pass {} has wrong results!, \nu={}\nv={}\ndiff={}'.format(
                __class__, ref_losses, check_losses, ref_losses - check_losses
            ),
        )

    def test_pp_pass(self):
        os.environ['FLAGS_new_executor_micro_batching'] = 'True'
        os.environ['FLAGS_enable_new_ir_in_executor'] = 'True'
        # navie pp without pipeline schedule
        engine_fthenb = self.get_engine(enable=False)
        outs = engine_fthenb.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert os.environ.get('FLAGS_new_executor_micro_batching') == "True"


if __name__ == "__main__":
    unittest.main()
