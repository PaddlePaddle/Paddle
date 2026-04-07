# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# Validates Muon optimizer with MuonShardingOptimizer.
# Muon requires whole 2D tensors for orthogonalization, so split_param is disabled.
# Tests all combinations of QKV/FFN/ns_coeff_type modes.
# Topology: sharding_degree=2, mp_degree=1 (2 ranks total)

import random
import unittest
from dataclasses import dataclass

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.optimizer.muon import (
    MuonParamInfo,
    QKVInfo,
    _default_should_use_muon,
)

# Parameter combinations
QKV_UPDATE_MODES = ["split_head", "split_qkv", "fused_qkv"]
FFN_SPLITS = [True, False]
NS_COEFF_TYPES = ["simple", "quintic", "polar_express", "aol"]

# Model config
vocab_size = 20
hidden_size = 64
head_num = 4  # num_attention_heads
kv_head_num = 2  # num_key_value_heads (GQA)
head_dim = hidden_size // head_num
intermediate_size = 128
qkv_dim = (head_num + 2 * kv_head_num) * head_dim
seq_length = 2
batch_size = 4
STEPS = 3

sharding_degree = 2


@dataclass
class TestConfig:
    """Test model config."""

    vocab_size: int = vocab_size
    hidden_size: int = hidden_size
    head_num: int = head_num
    kv_head_num: int = kv_head_num
    head_dim: int = head_dim
    intermediate_size: int = intermediate_size
    qkv_dim: int = qkv_dim


class QKVFFNNet(paddle.nn.Layer):
    """Test model with QKV and FFN gate_up.

    Parameter naming follows PaddleFormers:
    - qkv_proj.weight: QKV fused weights
    - up_gate_proj.weight: FFN gate_up fused weights
    """

    def __init__(
        self,
        config,
        np_embed,
        np_qkv,
        np_o_proj,
        np_up_gate,
        np_down_proj,
        np_lm_head,
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = paddle.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_embed)
            ),
        )

        self.qkv_proj = paddle.nn.Linear(
            config.hidden_size,
            config.qkv_dim,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_qkv)
            ),
        )

        self.o_proj = paddle.nn.Linear(
            config.head_num * config.head_dim,
            config.hidden_size,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_o_proj)
            ),
        )

        self.up_gate_proj = paddle.nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_up_gate)
            ),
        )

        self.down_proj = paddle.nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_down_proj)
            ),
        )

        self.lm_head = paddle.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias_attr=False,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.Assign(np_lm_head)
            ),
        )

    def forward(self, x):
        # Embedding
        h = self.embed_tokens(x)  # [batch, seq, hidden]

        # QKV projection (simplified: no real attention)
        qkv = self.qkv_proj(h)  # [batch, seq, qkv_dim]
        # Simplified: mean pooling to simulate attention output
        out = qkv.mean(axis=-1, keepdim=True)  # [batch, seq, 1]
        out = out.expand([x.shape[0], x.shape[1], self.config.hidden_size])

        # Output projection
        out = self.o_proj(out)  # [batch, seq, hidden]

        # FFN up_gate projection
        gate_up = self.up_gate_proj(out)  # [batch, seq, 2*intermediate]
        # Simplified: mean pooling
        out = gate_up.mean(axis=-1, keepdim=True)
        out = out.expand([x.shape[0], x.shape[1], self.config.hidden_size])

        # Down projection
        out = self.down_proj(out)  # [batch, seq, hidden]

        # LM head
        logits = self.lm_head(out)  # [batch, seq, vocab]

        return logits


class TestDistShardingMuonTraining(unittest.TestCase):
    def setUp(self):
        random.seed(2021)
        np.random.seed(2021)
        paddle.seed(2021)

        self.config = TestConfig()

        self.strategy = fleet.DistributedStrategy()
        self.strategy.hybrid_configs = {
            "sharding_degree": sharding_degree,
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        self.strategy.use_muon_sharding = True

        fleet.init(is_collective=True, strategy=self.strategy)
        self.data = [
            np.random.randint(0, vocab_size, (batch_size, seq_length))
            for _ in range(STEPS)
        ]

    def train_batch(self, batch, model, optimizer):
        with paddle.amp.auto_cast(dtype='bfloat16'):
            output = model(batch)
            loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return loss

    def build_optimizer(self, model, qkv_mode, ffn_split, ns_coeff):
        """Build Muon optimizer, ref: PaddleFormers trainer.py L3122-3173."""

        muon_param_info_map = {}
        exclude_patterns = ["embed", "bias", "lm_head"]

        for name, param in model.named_parameters():
            use_muon = _default_should_use_muon(
                name, param.shape, exclude_patterns
            )

            # QKV params: set QKVInfo
            if "qkv_proj.weight" in name and len(param.shape) == 2:
                param_info = MuonParamInfo(
                    use_muon=use_muon,
                    qkv_info=QKVInfo(
                        head_num=self.config.head_num,
                        kv_head_num=self.config.kv_head_num,
                        num_key_value_groups=self.config.head_num
                        // self.config.kv_head_num,
                    ),
                )
            # FFN gate_up params: set intermediate_size
            elif "up_gate_proj.weight" in name and ffn_split:
                param_info = MuonParamInfo(
                    use_muon=use_muon,
                    intermediate_size=self.config.intermediate_size,
                )
            else:
                param_info = MuonParamInfo(use_muon=use_muon)

            muon_param_info_map[param.name] = param_info
        return paddle.optimizer.Muon(
            parameters=model.parameters(),
            learning_rate=0.001,
            weight_decay=0.00001,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(0.5),
            muon_param_info_map=muon_param_info_map,
            muon_qkv_update_mode=qkv_mode,
            muon_ffn_split=ffn_split,
            ns_coeff_type=ns_coeff,
        )

    def _run_single_test(self, qkv_mode, ffn_split, ns_coeff):
        """Run single test combination."""
        # Init weights
        np_embed = np.random.random_sample((vocab_size, hidden_size))
        np_qkv = np.random.random_sample((hidden_size, qkv_dim))
        np_o_proj = np.random.random_sample((head_num * head_dim, hidden_size))
        np_up_gate = np.random.random_sample(
            (hidden_size, 2 * intermediate_size)
        )
        np_down_proj = np.random.random_sample((hidden_size, hidden_size))
        np_lm_head = np.random.random_sample((hidden_size, vocab_size))

        # Distributed model
        model_a = QKVFFNNet(
            self.config,
            np_embed,
            np_qkv,
            np_o_proj,
            np_up_gate,
            np_down_proj,
            np_lm_head,
        )
        model_a = mix_precision_utils.MixPrecisionLayer(
            model_a, dtype="bfloat16"
        )
        model_a = paddle.amp.decorate(
            models=model_a, level='O2', dtype='bfloat16'
        )
        optimizer_a = self.build_optimizer(
            model_a, qkv_mode, ffn_split, ns_coeff
        )

        # Single-GPU reference model (same MixPrecisionLayer pattern for consistency)
        model_b = QKVFFNNet(
            self.config,
            np_embed,
            np_qkv,
            np_o_proj,
            np_up_gate,
            np_down_proj,
            np_lm_head,
        )
        model_b = mix_precision_utils.MixPrecisionLayer(
            model_b, dtype="bfloat16"
        )
        model_b = paddle.amp.decorate(
            models=model_b, level='O2', dtype='bfloat16'
        )
        optimizer_b = self.build_optimizer(
            model_b, qkv_mode, ffn_split, ns_coeff
        )
        optimizer_b = mix_precision_utils.MixPrecisionOptimizer(optimizer_b)

        # Distributed wrapper
        model_a = fleet.distributed_model(model_a)
        optimizer_a = fleet.distributed_optimizer(optimizer_a)

        hcg = fleet.get_hybrid_communicate_group()
        sharding_rank = hcg.get_sharding_parallel_rank()
        local_batch_size = batch_size // sharding_degree

        for idx in range(STEPS):
            start = sharding_rank * local_batch_size
            batch_a = paddle.to_tensor(
                self.data[idx][start : start + local_batch_size]
            )
            batch_b = paddle.to_tensor(self.data[idx])

            loss_a = self.train_batch(batch_a, model_a, optimizer_a)
            loss_b = self.train_batch(batch_b, model_b, optimizer_b)

            # Verify param consistency
            for param_a, param_b in zip(
                model_a.parameters(), model_b.parameters()
            ):
                a_fp32 = param_a.cast('float32').numpy()
                b_fp32 = param_b.cast('float32').numpy()
                np.testing.assert_allclose(
                    a_fp32,
                    b_fp32,
                    atol=1e-3,
                    err_msg=f"Param {param_a.name} mismatch at step {idx}!",
                )

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8,
        "BF16 matmul requires GPU compute capability >= 80 (Ampere+)",
    )
    def test_sharding_muon(self):
        """Test all 24 parameter combinations."""
        total = len(QKV_UPDATE_MODES) * len(FFN_SPLITS) * len(NS_COEFF_TYPES)
        passed = 0
        failed = []

        for qkv_mode in QKV_UPDATE_MODES:
            for ffn_split in FFN_SPLITS:
                for ns_coeff in NS_COEFF_TYPES:
                    print(
                        f"\n[Muon Test] qkv_mode={qkv_mode}, ffn_split={ffn_split}, ns_coeff={ns_coeff}"
                    )
                    try:
                        self._run_single_test(qkv_mode, ffn_split, ns_coeff)
                        passed += 1
                        print(f"[PASS] {qkv_mode}, {ffn_split}, {ns_coeff}")
                    except Exception as e:
                        failed.append((qkv_mode, ffn_split, ns_coeff, str(e)))
                        print(
                            f"[FAIL] {qkv_mode}, {ffn_split}, {ns_coeff}: {e}"
                        )

        print(f"\n{'=' * 60}")
        print(f"Muon Sharding Test Summary: {passed}/{total} passed")
        if failed:
            print("Failed combinations:")
            for qkv, ffn, ns, err in failed:
                print(f"  - {qkv}, {ffn}, {ns}: {err[:100]}...")
        print(f"{'=' * 60}")

        if failed:
            raise AssertionError(f"{len(failed)} test combinations failed")


if __name__ == "__main__":
    unittest.main()
