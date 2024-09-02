# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest

import numpy as np

sys.path.append("../../auto_parallel")

from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass(use_zbh1=False, enable_send_recv_overlap=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    if use_zbh1:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = "ZBH1"
        pipeline.accumulate_steps = 2
        pipeline.enable_send_recv_overlap = enable_send_recv_overlap
    else:
        gradient_merge = strategy.gradient_merge
        gradient_merge.enable = True
        gradient_merge.k_steps = 2
        gradient_merge.avg = True

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class TestZBH1Pass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 4
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.base.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_zbh1=False, enable_send_recv_overlap=False):
        reset_prog()

        strategy = apply_pass(use_zbh1, enable_send_recv_overlap)

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
            err_msg=f'pass {__class__} has wrong results!, \nu={ref_losses}\nv={check_losses}\ndiff={ref_losses - check_losses}',
        )

    def test_pp_pass(self):
        # naive_pp+gradient_merge training
        engine_pp = self.get_engine()
        history_pp = engine_pp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_pp._strategy.pipeline.enable is False

        # pp2 zbh1 training
        engine_zbh1 = self.get_engine(True)
        history_zbh1 = engine_zbh1.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_zbh1._strategy.pipeline.enable is True

        # NOTE: every sample data from dataset is all the same
        if paddle.distributed.get_rank() == 1:
            losses_pp = np.array(history_pp.history["loss"])
            losses_zbh1 = np.array(history_zbh1.history["loss"])
            self.check_results(losses_pp[0], losses_zbh1[0])

    def test_pp_pass_enable_send_recv_overlap(self):
        # naive_pp+gradient_merge training
        engine_pp = self.get_engine(enable_send_recv_overlap=True)
        history_pp = engine_pp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_pp._strategy.pipeline.enable is False

        # pp2 zbh1 training
        engine_zbh1 = self.get_engine(True, enable_send_recv_overlap=True)
        history_zbh1 = engine_zbh1.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_zbh1._strategy.pipeline.enable is True

        # NOTE: every sample data from dataset is all the same
        if paddle.distributed.get_rank() == 1:
            losses_pp = np.array(history_pp.history["loss"])
            losses_zbh1 = np.array(history_zbh1.history["loss"])
            self.check_results(losses_pp[0], losses_zbh1[0])


if __name__ == "__main__":
    unittest.main()
