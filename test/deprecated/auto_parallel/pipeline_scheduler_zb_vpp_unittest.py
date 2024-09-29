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
from get_gpt_model import FakeDataset, generate_model

import paddle
from paddle.distributed import ParallelEnv
from paddle.distributed.fleet import auto

sys.path.append("../legacy_test")

paddle.enable_static()


def apply_pass(use_zbvpp=False, enable_send_recv_overlap=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    if use_zbvpp:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = "ZBVPP"
        pipeline.accumulate_steps = 2
        pipeline.vpp_degree = 3
        pipeline.vpp_seg_method = "TransformerDecoderLayer"
        pipeline.enable_send_recv_overlap = enable_send_recv_overlap
    else:
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = "VPP"
        pipeline.accumulate_steps = 2
        pipeline.vpp_degree = 3
        pipeline.vpp_seg_method = "TransformerDecoderLayer"
        pipeline.enable_send_recv_overlap = enable_send_recv_overlap

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())


class TestZBVPPPass(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-2
        self.atol = 1e-3
        self.batch_size = 2
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

    def get_engine(self, use_zbvpp=False, enable_send_recv_overlap=False):
        reset_prog()

        strategy = apply_pass(use_zbvpp, enable_send_recv_overlap)

        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("pp", num_hidden_layers=6)

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
        # pp2 vpp training
        engine_vpp = self.get_engine(False)
        history_vpp = engine_vpp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_vpp._strategy.pipeline.enable is True

        # pp2 zbvpp training
        engine_zbvpp = self.get_engine(True)
        history_zbvpp = engine_zbvpp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_zbvpp._strategy.pipeline.enable is True

        if paddle.distributed.get_rank() == 1:
            losses_vpp = np.array(history_vpp.history["loss"])
            history_zbvpp = np.array(history_zbvpp.history["loss"])
            self.check_results(losses_vpp[0], history_zbvpp[0])

    def test_pp_pass_enable_send_recv_overlap(self):
        # pp2 vpp training
        engine_vpp = self.get_engine(False, enable_send_recv_overlap=True)
        history_vpp = engine_vpp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_vpp._strategy.pipeline.enable is True

        # pp2 zbvpp training
        engine_zbvpp = self.get_engine(True, enable_send_recv_overlap=True)
        history_zbvpp = engine_zbvpp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_zbvpp._strategy.pipeline.enable is True

        if paddle.distributed.get_rank() == 1:
            losses_vpp = np.array(history_vpp.history["loss"])
            history_zbvpp = np.array(history_zbvpp.history["loss"])
            self.check_results(losses_vpp[0], history_zbvpp[0])


if __name__ == "__main__":
    unittest.main()
