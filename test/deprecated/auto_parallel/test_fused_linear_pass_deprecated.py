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
import sys
import unittest

import numpy as np

sys.path.append("../../auto_parallel")
sys.path.append("../../legacy_test")
from get_gpt_model import FakeDataset, generate_model
from test_sparse_addmm_op import get_cuda_version

import paddle
from paddle.distributed.fleet import auto


def apply_pass(use_fused_passes=False, fused_passes_list=[]):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True
    fused_passes = strategy.fused_passes
    fused_passes.enable = use_fused_passes
    fused_passes.fused_passes_list = fused_passes_list
    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class TestFusedLinearPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        self.atol = 1e-8
        self.batch_size = 1
        self.batch_num = 1
        self.clip_norm = 0.2
        self.dataset = FakeDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_fused_passes=False, fused_passes_list=[]):
        reset_prog()

        strategy = apply_pass(use_fused_passes, fused_passes_list)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("serial")

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_results(self, ref_losses, check_losses, rtol=None, atol=None):
        np.testing.assert_allclose(
            ref_losses,
            check_losses,
            rtol=rtol or self.rtol,
            atol=atol or self.atol,
            err_msg=f'pass {__class__} has wrong results!, \nu={ref_losses}\nv={check_losses}\ndiff={ref_losses - check_losses}',
        )

    def test_passes(self):
        losses = []
        if get_cuda_version() >= 11060:
            for use_fused_passes in [True, False]:
                engine = self.get_engine(
                    use_fused_passes, ["fuse_gemm_epilogue"]
                )
                history = engine.fit(
                    self.dataset, 3, batch_size=self.batch_size
                )
                losses.append(np.array(history.history["loss"]))
            self.check_results(losses[0], losses[1])


if __name__ == "__main__":
    unittest.main()
