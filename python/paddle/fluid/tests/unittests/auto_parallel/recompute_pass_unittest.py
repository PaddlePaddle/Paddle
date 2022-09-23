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

import unittest
import sys
import random
import numpy as np
import paddle

from paddle.distributed.fleet import auto
from paddle.fluid.dygraph.parallel import ParallelEnv
from get_gpt_model import generate_model, create_data_holder, FakeDataset


def apply_pass(use_recompute=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    if use_recompute:
        recompute = strategy.recompute
        recompute.enable = True
    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestRecomputePass(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-6
        self.atol = 1e-8
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_recompute=False):
        reset_prog()

        strategy = apply_pass(use_recompute)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("mp")

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
                __class__, ref_losses, check_losses, ref_losses - check_losses))

    def test_recompute_pass(self):
        # mp2 training
        mp_engine = self.get_engine()
        mp_losses = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        mp_losses = np.array(mp_losses["loss"])

        # mp2 recompute training
        rc_engine = self.get_engine(True)
        rc_losses = rc_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        rc_losses = np.array(rc_losses["loss"])
        self.check_results(mp_losses, rc_losses)


if __name__ == "__main__":
    unittest.main()
