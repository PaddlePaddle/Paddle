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
from paddle.distributed.auto_parallel.static.utils import (
    is_backward_op,
    is_forward_op,
    is_optimize_op,
)
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
        pipeline.vpp_degree = 2
        pipeline.vpp_seg_method = "TransformerDecoderLayer"
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


class TestZBVPPPass(unittest.TestCase):
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
        place = paddle.base.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_zbvpp=False, enable_send_recv_overlap=False):
        reset_prog()

        strategy = apply_pass(use_zbvpp, enable_send_recv_overlap)

        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model, loss = generate_model("pp", num_hidden_layers=4)

        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def test_pp_pass(self):
        # pp2 zbvpp training
        engine_zbvpp = self.get_engine(True)
        history_zbvpp = engine_zbvpp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_zbvpp._strategy.pipeline.enable is True

        fw_chunk_ids = []
        bw_chunk_ids = []
        for op in engine_zbvpp.main_program.global_block().ops:
            if is_optimize_op(op):
                break

            dist_op = engine_zbvpp.dist_context.get_dist_op_for_program(op)
            if is_forward_op(op):
                fw_chunk_ids.append(dist_op.dist_attr.chunk_id)
            if is_backward_op(op):
                bw_chunk_ids.append(dist_op.dist_attr.chunk_id)

        if paddle.distributed.get_rank() == 0:
            self.assertEqual(sum(fw_chunk_ids), 43)
            self.assertEqual(sum(bw_chunk_ids), 60)
        else:
            self.assertEqual(sum(fw_chunk_ids), 32)
            self.assertEqual(sum(bw_chunk_ids), 50)

    def test_pp_pass_enable_send_recv_overlap(self):
        # pp2 zbvpp training
        engine_zbvpp = self.get_engine(True, enable_send_recv_overlap=True)
        history_zbvpp = engine_zbvpp.fit(
            self.dataset, 3, batch_size=self.batch_size, log_freq=1
        )
        assert engine_zbvpp._strategy.pipeline.enable is True

        fw_chunk_ids = []
        bw_chunk_ids = []
        for op in engine_zbvpp.main_program.global_block().ops:
            if is_optimize_op(op):
                break

            dist_op = engine_zbvpp.dist_context.get_dist_op_for_program(op)
            if is_forward_op(op):
                fw_chunk_ids.append(dist_op.dist_attr.chunk_id)
            if is_backward_op(op):
                bw_chunk_ids.append(dist_op.dist_attr.chunk_id)

        if paddle.distributed.get_rank() == 0:
            self.assertEqual(sum(fw_chunk_ids), 43)
            self.assertEqual(sum(bw_chunk_ids), 60)
        else:
            self.assertEqual(sum(fw_chunk_ids), 32)
            self.assertEqual(sum(bw_chunk_ids), 50)


if __name__ == "__main__":
    unittest.main()
