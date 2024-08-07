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
from paddle.distributed.fleet import auto

paddle.enable_static()


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class TestMPAllreduceMatmulGradOverlapping(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 1
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2023)
        np.random.seed(2023)
        random.seed(2023)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_mp_engine(self, allreduce_matmul_grad_overlapping):
        reset_prog()

        strategy = auto.Strategy()
        strategy.auto_mode = "semi"
        strategy.reinit = True
        strategy.mp_optimization.allreduce_matmul_grad_overlapping = (
            allreduce_matmul_grad_overlapping
        )

        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("mp")

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def run_mp(self, allreduce_matmul_grad_overlapping):
        mp_engine = self.get_mp_engine(allreduce_matmul_grad_overlapping)
        history = mp_engine.fit(self.dataset, 3, batch_size=self.batch_size)
        return np.array(history.history["loss"])

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=rtol or self.rtol,
            atol=atol or self.atol,
            err_msg=f'pass {__class__} has wrong results!, \nu={ref_losses}\nv={check_losses}\ndiff={ref_losses - check_losses}',
        )

    def test_mp_allreduce_matmul_grad_overlapping(self):
        losses_with_allreduce_matmul_grad_overlapping = self.run_mp(True)
        losses_without_allreduce_matmul_grad_overlapping = self.run_mp(False)

        np.testing.assert_equal(
            losses_with_allreduce_matmul_grad_overlapping,
            losses_without_allreduce_matmul_grad_overlapping,
        )


if __name__ == "__main__":
    unittest.main()
