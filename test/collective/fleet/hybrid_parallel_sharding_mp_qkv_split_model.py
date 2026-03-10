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
#
# Validates Muon is_split_qkv=True: each Q/K/V head is orthogonalised
# independently rather than the whole Q/K/V block at once.
#
# Topology: embedding -> fused qkv_proj -> out_proj -> lm_head
# Parallelism: mp_degree=2, sharding_degree=1 (2 ranks total)

import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionLayer,
    MixPrecisionOptimizer,
)

# Model hyper-parameters
vocab_size = 32
hidden_size = 64
head_num = 4  # Q heads
kv_head_num = 2  # K/V heads (GQA)
head_dim = hidden_size // head_num
qkv_out_dim = (head_num + 2 * kv_head_num) * head_dim
output_size = vocab_size
seq_length = 4
batch_size = 4
STEPS = 3

sharding_degree = 1
mp_degree = 2


class SimpleMPQKVNet(paddle.nn.Layer):
    """TP model: column-parallel qkv_proj, row-parallel out_proj."""

    def __init__(self, np_emb, np_qkv, np_out, np_lm):
        super().__init__()

        hcg = fleet.get_hybrid_communicate_group()
        mp_id = hcg.get_model_parallel_rank()
        mp_deg = hcg.get_model_parallel_world_size()

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_emb)
            ),
        )

        qkv_per_rank = qkv_out_dim // mp_deg
        col_start = mp_id * qkv_per_rank
        col_end = col_start + qkv_per_rank
        self.qkv_proj = fleet.meta_parallel.ColumnParallelLinear(
            hidden_size,
            qkv_out_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    np_qkv[:, col_start:col_end]
                ),
            ),
            gather_output=False,
            has_bias=False,
        )

        self.out_proj = fleet.meta_parallel.RowParallelLinear(
            qkv_out_dim,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    np_out[col_start:col_end, :]
                ),
            ),
            input_is_parallel=True,
            has_bias=False,
        )

        self.lm_head = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_lm)
            ),
            bias_attr=False,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.qkv_proj(x)
        x = self.out_proj(x)
        x = self.lm_head(x)
        return x


class SimpleDPQKVNet(paddle.nn.Layer):
    """Single-rank reference model with identical weight initialisation."""

    def __init__(self, np_emb, np_qkv, np_out, np_lm):
        super().__init__()

        self.embedding = paddle.nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_emb)
            ),
        )

        self.qkv_proj = paddle.nn.Linear(
            hidden_size,
            qkv_out_dim,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_qkv),
            ),
            bias_attr=False,
        )

        self.out_proj = paddle.nn.Linear(
            qkv_out_dim,
            hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_out)
            ),
            bias_attr=False,
        )

        self.lm_head = paddle.nn.Linear(
            hidden_size,
            output_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_lm)
            ),
            bias_attr=False,
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.qkv_proj(x)
        x = self.out_proj(x)
        x = self.lm_head(x)
        return x


class TestDistMPQKVSplitTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2024)
        np.random.seed(2024)
        paddle.seed(2024)

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": sharding_degree,
            "dp_degree": 1,
            "mp_degree": mp_degree,
            "pp_degree": 1,
        }

        fleet.init(is_collective=True, strategy=self.strategy)

        self.data = [
            np.random.randint(0, vocab_size, (batch_size, seq_length))
            for _ in range(STEPS)
        ]

    def _random_weights(self):
        np_emb = np.random.random_sample((vocab_size, hidden_size)).astype(
            "float32"
        )
        np_qkv = np.random.random_sample((hidden_size, qkv_out_dim)).astype(
            "float32"
        )
        np_out = np.random.random_sample((qkv_out_dim, hidden_size)).astype(
            "float32"
        )
        np_lm = np.random.random_sample((hidden_size, output_size)).astype(
            "float32"
        )
        return np_emb, np_qkv, np_out, np_lm

    def _mark_qkv_params(self, model):
        """Set needs_qkv_split / head_num / kv_head_num on every qkv_proj weight."""
        for name, param in model.named_parameters():
            if "qkv_proj" in name and param.ndim == 2:
                param.needs_qkv_split = True
                param.head_num = head_num
                param.kv_head_num = kv_head_num

    def _build_muon_optimizer(self, model):
        return paddle.optimizer.Muon(
            parameters=model.parameters(),
            learning_rate=0.001,
            weight_decay=0.00001,
            is_split_qkv=True,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(0.5),
        )

    def _train_batch(self, batch, model, optimizer):
        loss = model(batch).mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return loss

    def _run_split_qkv(self, amp_level=None):
        np_emb, np_qkv, np_out, np_lm = self._random_weights()

        model_a = SimpleMPQKVNet(np_emb, np_qkv, np_out, np_lm)
        self._mark_qkv_params(model_a)
        optimizer_a = self._build_muon_optimizer(model_a)

        model_b = SimpleDPQKVNet(np_emb, np_qkv, np_out, np_lm)
        self._mark_qkv_params(model_b)
        optimizer_b = self._build_muon_optimizer(model_b)

        if amp_level == "O2":
            model_a = MixPrecisionLayer(model_a)
            optimizer_a = MixPrecisionOptimizer(optimizer_a)
            model_b = MixPrecisionLayer(model_b)
            optimizer_b = MixPrecisionOptimizer(optimizer_b)

        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()

        for idx in range(STEPS):
            batch = paddle.to_tensor(self.data[idx])

            loss_a = self._train_batch(batch, model_a, optimizer_a)
            loss_b = self._train_batch(batch, model_b, optimizer_b)

            for param_a, param_b in zip(
                model_a.parameters(), model_b.parameters()
            ):
                val_a_local = param_a.numpy()
                val_b = param_b.numpy()

                if val_a_local.shape != val_b.shape:
                    gathered = []
                    paddle.distributed.all_gather(
                        gathered, param_a, group=tp_group
                    )
                    concat_axis = next(
                        (
                            d
                            for d in range(len(val_b.shape))
                            if val_b.shape[d]
                            == val_a_local.shape[d] * mp_degree
                        ),
                        -1,
                    )
                    if concat_axis == -1:
                        continue
                    val_a_global = np.concatenate(
                        [t.numpy() for t in gathered], axis=concat_axis
                    )
                else:
                    val_a_global = val_a_local

                np.testing.assert_allclose(
                    val_a_global,
                    val_b,
                    rtol=1e-4,
                    atol=3e-4,
                    err_msg=f"Param {param_a.name} mismatch at step {idx}!",
                )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8,
        "BF16 matmul requires GPU compute capability >= 80 (Ampere+)",
    )
    def test_muon_split_qkv_per_head(self):
        """Muon per-head QKV split: TP only, BF16 AMP O2."""
        self._run_split_qkv(amp_level="O2")


if __name__ == "__main__":
    unittest.main()
