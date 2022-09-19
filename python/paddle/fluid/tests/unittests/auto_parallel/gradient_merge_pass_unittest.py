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

paddle.enable_static()


def apply_pass(use_gradient_merge=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    if use_gradient_merge:
        gradient_merge = strategy.gradient_merge
        gradient_merge.enable = True
        gradient_merge.k_steps = 4
        gradient_merge.avg = True

    return strategy


def reset_prog():
    paddle.fluid.framework.switch_main_program(paddle.static.Program())
    paddle.fluid.framework.switch_startup_program(paddle.static.Program())


class TestGradientMergePass(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 8
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.fluid.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_gradient_merge=False):
        reset_prog()

        strategy = apply_pass(use_gradient_merge)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("dp")

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

    def test_gradient_merge_pass(self):
        # dp2 training
        dp_engine = self.get_engine()
        dp_losses = dp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        dp_losses = np.array(dp_losses["loss"])

        # dp2 gradient merge training
        gm_engine = self.get_engine(True)
        gm_losses = gm_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        gm_losses = np.array(gm_losses["loss"])

        avg_loss = 0
        pass_avg_ret_list = []
        for i, pass_ret in enumerate(gm_losses):
            if (i + 1) % 4 == 0:
                avg_loss += pass_ret
                pass_avg_ret_list.append(avg_loss / 4)
                avg_loss = 0
            else:
                avg_loss += pass_ret

        self.check_results(dp_losses, np.array(pass_avg_ret_list))


if __name__ == "__main__":
    unittest.main()
