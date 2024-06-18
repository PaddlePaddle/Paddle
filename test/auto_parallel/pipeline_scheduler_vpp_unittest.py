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

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import ParallelEnv
from paddle.distributed.auto_parallel.static.utils import (
    is_backward_op,
    is_forward_op,
    is_optimize_op,
)
from paddle.distributed.fleet import auto

paddle.enable_static()

PP_MESH_0 = auto.ProcessMesh([0])
PP_MESH_1 = auto.ProcessMesh([1])


class MyLinear(nn.Layer):
    def __init__(
        self,
        hidden_size=784,
        intermediate_size=4 * 784,
        dropout_ratio=0.1,
        weight_attr=None,
    ):
        super().__init__()

        self.linear0 = nn.Linear(
            hidden_size, intermediate_size, weight_attr, bias_attr=None
        )
        self.linear1 = nn.Linear(
            intermediate_size, hidden_size, weight_attr, bias_attr=None
        )
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = self.linear0(input)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)

        return out


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=784,
        intermediate_size=4 * 784,
        dropout_ratio=0.1,
        initializer_range=0.02,
        manual=True,
    ):
        super().__init__()

        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )

        self.layers = nn.LayerList(
            [
                MyLinear(
                    hidden_size, intermediate_size, dropout_ratio, weight_attr
                )
                for _ in range(4)
            ]
        )

        self.linear = nn.Linear(hidden_size, 1, weight_attr, bias_attr=None)
        self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
        if manual:
            self.layer_to_mesh = [PP_MESH_0, PP_MESH_1, PP_MESH_0, PP_MESH_1]
        else:
            self.layer_to_mesh = [PP_MESH_0, PP_MESH_0, PP_MESH_1, PP_MESH_1]

    def forward(self, input):
        out = self.norm(input)

        for i, layer in enumerate(self.layers):
            auto.shard_tensor(out, self.layer_to_mesh[i], [None, None])
            out = layer(out)

        out = self.linear(out)
        return out


def loss_fn(pred, label):
    loss = F.l1_loss(pred, label)
    return loss


def apply_pass(schedule_mode, acc_step, enable_send_recv_overlap=False):
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"
    strategy.reinit = True

    pipeline = strategy.pipeline
    pipeline.enable = True
    pipeline.schedule_mode = schedule_mode
    pipeline.accumulate_steps = acc_step
    pipeline.vpp_degree = 2
    pipeline.vpp_seg_method = "MyLinear"
    pipeline.enable_send_recv_overlap = enable_send_recv_overlap

    return strategy


def reset_prog():
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())
    paddle.utils.unique_name.switch()


class MyDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=784).astype("float32")
        label = np.random.uniform(size=1).astype("float32")
        return input, label

    def __len__(self):
        return self.num_samples


class TestVPPPass(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.batch_num = 10
        self.clip_norm = 0.2
        self.dataset = MyDataset(self.batch_size * self.batch_num)

    def init(self, engine):
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        paddle.distributed.fleet.init(is_collective=True)
        place = paddle.base.CUDAPlace(ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(
        self,
        schedule_mode,
        acc_step,
        manual=True,
        enable_send_recv_overlap=False,
    ):
        reset_prog()

        strategy = apply_pass(schedule_mode, acc_step, enable_send_recv_overlap)
        clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
        opt = paddle.optimizer.AdamW(learning_rate=0.00001, grad_clip=clip)
        model = MLPLayer(manual=manual)

        engine = auto.Engine(model, loss_fn, opt, strategy=strategy)
        self.init(engine)
        return engine

    def test_pp_pass(self):
        # pp2-vpp-manual
        engine = self.get_engine(schedule_mode="VPP", acc_step=4, manual=True)
        out_manual = engine.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine._strategy.pipeline.schedule_mode == "VPP"

        fw_chunk_ids = []
        bw_chunk_ids = []
        for op in engine.main_program.global_block().ops:
            if is_optimize_op(op):
                break

            dist_op = engine.dist_context.get_dist_op_for_program(op)
            if is_forward_op(op):
                fw_chunk_ids.append(dist_op.dist_attr.chunk_id)
            if is_backward_op(op):
                bw_chunk_ids.append(dist_op.dist_attr.chunk_id)

        if paddle.distributed.get_rank() == 0:
            self.assertEqual(sum(fw_chunk_ids), 8)
            self.assertEqual(sum(bw_chunk_ids), 13)
        else:
            self.assertEqual(sum(fw_chunk_ids), 12)
            self.assertEqual(sum(bw_chunk_ids), 19)

        # pp2-vpp-auto
        engine = self.get_engine(schedule_mode="VPP", acc_step=4, manual=False)
        out_auto = engine.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine._strategy.pipeline.schedule_mode == "VPP"

        fw_chunk_ids = []
        bw_chunk_ids = []
        for op in engine.main_program.global_block().ops:
            if is_optimize_op(op):
                break

            dist_op = engine.dist_context.get_dist_op_for_program(op)
            if is_forward_op(op):
                fw_chunk_ids.append(dist_op.dist_attr.chunk_id)
            if is_backward_op(op):
                bw_chunk_ids.append(dist_op.dist_attr.chunk_id)

        if paddle.distributed.get_rank() == 0:
            self.assertEqual(sum(fw_chunk_ids), 9)
            self.assertEqual(sum(bw_chunk_ids), 13)
        else:
            self.assertEqual(sum(fw_chunk_ids), 13)
            self.assertEqual(sum(bw_chunk_ids), 19)

        # pp2-vpp-manual-overlap
        engine = self.get_engine(
            schedule_mode="VPP",
            acc_step=4,
            manual=True,
            enable_send_recv_overlap=True,
        )
        out_manual_overlap = engine.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine._strategy.pipeline.schedule_mode == "VPP"
        assert engine._strategy.pipeline.enable_send_recv_overlap is True

        fw_chunk_ids = []
        bw_chunk_ids = []
        for op in engine.main_program.global_block().ops:
            if is_optimize_op(op):
                break

            dist_op = engine.dist_context.get_dist_op_for_program(op)
            if is_forward_op(op):
                fw_chunk_ids.append(dist_op.dist_attr.chunk_id)
            if is_backward_op(op):
                bw_chunk_ids.append(dist_op.dist_attr.chunk_id)

        if paddle.distributed.get_rank() == 0:
            self.assertEqual(sum(fw_chunk_ids), 8)
            self.assertEqual(sum(bw_chunk_ids), 13)
        else:
            self.assertEqual(sum(fw_chunk_ids), 12)
            self.assertEqual(sum(bw_chunk_ids), 19)

        # pp2-vpp-auto-overlap
        engine = self.get_engine(
            schedule_mode="VPP",
            acc_step=4,
            manual=False,
            enable_send_recv_overlap=True,
        )
        out_auto_overlap = engine.fit(
            self.dataset, batch_size=self.batch_size, log_freq=1
        )
        assert engine._strategy.pipeline.schedule_mode == "VPP"
        assert engine._strategy.pipeline.enable_send_recv_overlap is True

        fw_chunk_ids = []
        bw_chunk_ids = []

        for op in engine.main_program.global_block().ops:
            if is_optimize_op(op):
                break

            dist_op = engine.dist_context.get_dist_op_for_program(op)
            if is_forward_op(op):
                fw_chunk_ids.append(dist_op.dist_attr.chunk_id)
            if is_backward_op(op):
                bw_chunk_ids.append(dist_op.dist_attr.chunk_id)

        if paddle.distributed.get_rank() == 0:
            self.assertEqual(sum(fw_chunk_ids), 9)
            self.assertEqual(sum(bw_chunk_ids), 13)
        else:
            self.assertEqual(sum(fw_chunk_ids), 13)
            self.assertEqual(sum(bw_chunk_ids), 19)

        if paddle.distributed.get_rank() == 1:
            self.assertEqual(
                np.mean(out_manual.history["loss"][0]),
                np.mean(out_auto.history["loss"][0]),
            )
            self.assertEqual(
                np.mean(out_manual.history["loss"][0]),
                np.mean(out_manual_overlap.history["loss"][0]),
            )
            self.assertEqual(
                np.mean(out_manual.history["loss"][0]),
                np.mean(out_auto_overlap.history["loss"][0]),
            )


if __name__ == "__main__":
    unittest.main()
