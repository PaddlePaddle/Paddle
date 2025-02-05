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

import random
import unittest

import numpy as np
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

paddle.enable_static()


def apply_pass(schedule_mode="FThenB", enable_send_recv_overlap=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    pipeline = strategy.pipeline
    pipeline.enable = True
    pipeline.schedule_mode = schedule_mode
    pipeline.accumulate_steps = 4
    pipeline.enable_send_recv_overlap = enable_send_recv_overlap

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class Test1F1BPass(unittest.TestCase):
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

    def get_engine(
        self, schedule_mode="FThenB", enable_send_recv_overlap=False
    ):
        reset_prog()

        strategy = apply_pass(schedule_mode, enable_send_recv_overlap)
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
        # pp2 fthenb training with standalone executor
        engine_fthenb = self.get_engine(schedule_mode="FThenB")
        history_fthenb = engine_fthenb.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_fthenb._strategy.pipeline.schedule_mode == "FThenB"

        # pp2 1f1b training with standalone executor
        engine_1f1b = self.get_engine(schedule_mode="1F1B")
        history_1f1b = engine_1f1b.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_1f1b._strategy.pipeline.schedule_mode == "1F1B"

        # pp2 eager1f1b training with standalone executor
        engine_eager1f1b = self.get_engine(schedule_mode="Eager1F1B")
        history_eager1f1b = engine_eager1f1b.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_eager1f1b._strategy.pipeline.schedule_mode == "Eager1F1B"

        # pp2 1f1b training with standalone executor
        engine_1f1b_overlap = self.get_engine(
            schedule_mode="1F1B", enable_send_recv_overlap=True
        )
        history_1f1b_overlap = engine_1f1b_overlap.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_1f1b_overlap._strategy.pipeline.schedule_mode == "1F1B"
        assert (
            engine_1f1b_overlap._strategy.pipeline.enable_send_recv_overlap
            is True
        )

        # NOTE: every sample data from dataset is all the same
        if paddle.distributed.get_rank() == 1:
            losses_fthenb = np.array(history_fthenb.history["loss"])
            losses_1f1b = np.array(history_1f1b.history["loss"])
            losses_eager1f1b = np.array(history_eager1f1b.history["loss"])
            losses_1f1b_overlap = np.array(history_1f1b_overlap.history["loss"])
            # accumulate_steps is 4
            assert losses_fthenb[0].shape[0] == 4
            assert losses_1f1b[0].shape[0] == 4
            assert losses_eager1f1b[0].shape[0] == 4
            assert losses_1f1b_overlap[0].shape[0] == 4


if __name__ == "__main__":
    unittest.main()
