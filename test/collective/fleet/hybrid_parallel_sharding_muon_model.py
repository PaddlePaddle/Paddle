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
# Tests ns_coeff_type modes, custom color groups, and split_concat_func.
# Topology: sharding_degree=2, mp_degree=1 (2 ranks total)

import os
import random
import unittest
from dataclasses import dataclass
from functools import partial

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils
from paddle.optimizer.muon import (
    MuonParamInfo,
    _default_should_use_muon,
)

# Enable MUON_DEBUG to cover the debug logging branch (muon.py L532-539)
os.environ["MUON_DEBUG"] = "1"

# Test-controlled flags (set via need_envs from test_parallel_dygraph_muon.py)
g_enable_fuse_optimizer_states = int(
    os.environ.get("ENABLE_FUSE_OPTIMIZER_STATES", "0")
)
g_release_gradients = int(os.environ.get("RELEASE_GRADIENTS", "0"))
g_multi_precision = int(os.environ.get("MULTI_PRECISION", "0"))

# Parameter combinations
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


# ------------------------------------------------------------------
# Slice functions (called as split_concat_func(matrix_2d_global, ortho_fn))
# ------------------------------------------------------------------


def _qkv_sep(matrix_2d, ortho_fn, kv_head_num=None, num_key_value_groups=None):
    """Slice QKV into Q, K, V blocks, orthogonalise each as whole."""
    head_dim_local = matrix_2d.shape[1] // (
        num_key_value_groups * kv_head_num + 2 * kv_head_num
    )
    q_dim = num_key_value_groups * kv_head_num * head_dim_local
    k_dim = kv_head_num * head_dim_local
    v_dim = kv_head_num * head_dim_local

    q, k, v = paddle.split(matrix_2d, [q_dim, k_dim, v_dim], axis=1)
    return paddle.concat([ortho_fn(q), ortho_fn(k), ortho_fn(v)], axis=1)


def _ffn_split(matrix_2d, ortho_fn, intermediate_size=None):
    """Split gate_up into gate and up, orthogonalise each."""
    gate, up = paddle.split(
        matrix_2d, [intermediate_size, intermediate_size], axis=1
    )
    return paddle.concat([ortho_fn(gate), ortho_fn(up)], axis=1)


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

        # Configure sharding_configs from env vars
        if g_enable_fuse_optimizer_states:
            self.strategy.hybrid_configs[
                "sharding_configs"
            ].enable_fuse_optimizer_states = True
        if g_release_gradients:
            self.strategy.hybrid_configs[
                "sharding_configs"
            ].release_gradients = True

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

    def _init_weights(self):
        """Create shared numpy weight arrays."""
        return (
            np.random.random_sample((vocab_size, hidden_size)),
            np.random.random_sample((hidden_size, qkv_dim)),
            np.random.random_sample((head_num * head_dim, hidden_size)),
            np.random.random_sample((hidden_size, 2 * intermediate_size)),
            np.random.random_sample((hidden_size, hidden_size)),
            np.random.random_sample((hidden_size, vocab_size)),
        )

    def _build_split_concat_func_map(self, model):
        """Build split_concat_func_map with QKV sep and FFN split for applicable params."""
        num_key_value_groups = head_num // kv_head_num
        slice_map = {}
        for name, param in model.named_parameters():
            if "qkv_proj" in name:
                slice_map[param.name] = partial(
                    _qkv_sep,
                    kv_head_num=kv_head_num,
                    num_key_value_groups=num_key_value_groups,
                )
            elif "up_gate_proj" in name:
                slice_map[param.name] = partial(
                    _ffn_split,
                    intermediate_size=intermediate_size,
                )
        return slice_map

    def build_optimizer(
        self,
        model,
        ns_coeff,
        split_concat_func_map=None,
        ns_matmul_dtype=None,
        multi_precision=False,
        apply_decay_param_fun=None,
    ):
        """Build Muon optimizer.

        Args:
            model: The model to optimize.
            ns_coeff: Newton-Schulz coefficient type.
            split_concat_func_map: Optional dict {param.name: split_concat_func}.
                Covers muon.py L529 (split_concat_func call) and L535 (debug log).
            ns_matmul_dtype: Optional explicit dtype for NS matmul.
                Covers muon.py L283 (explicit ns_matmul_dtype branch).
            multi_precision: If True, enable FP32 master weights.
                Covers muon.py L560-564, L574-575, L582-583.
            apply_decay_param_fun: Optional callable(param_name) -> bool.
                Covers muon.py L443-446, L568-572.
        """
        muon_param_info_map = {}
        exclude_patterns = ["embed", "bias", "lm_head"]

        for name, param in model.named_parameters():
            use_muon = _default_should_use_muon(
                name, param.shape, exclude_patterns
            )
            sf = None
            if split_concat_func_map and param.name in split_concat_func_map:
                sf = split_concat_func_map[param.name]
            param_info = MuonParamInfo(use_muon=use_muon, split_concat_func=sf)
            muon_param_info_map[param.name] = param_info

        kwargs = {}
        if ns_matmul_dtype is not None:
            kwargs['ns_matmul_dtype'] = ns_matmul_dtype

        return paddle.optimizer.Muon(
            parameters=model.parameters(),
            learning_rate=0.001,
            weight_decay=0.00001,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(0.5),
            muon_param_info_map=muon_param_info_map,
            ns_coeff_type=ns_coeff,
            multi_precision=multi_precision,
            apply_decay_param_fun=apply_decay_param_fun,
            **kwargs,
        )

    def _build_model(self, weights):
        """Build a single model instance from weights."""
        np_embed, np_qkv, np_o_proj, np_up_gate, np_down_proj, np_lm_head = (
            weights
        )
        model = QKVFFNNet(
            self.config,
            np_embed,
            np_qkv,
            np_o_proj,
            np_up_gate,
            np_down_proj,
            np_lm_head,
        )
        model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
        model = paddle.amp.decorate(models=model, level='O2', dtype='bfloat16')
        return model

    def _run_single_test(
        self,
        ns_coeff,
        color_params=None,
        use_slice=False,
        explicit_dtype=False,
        multi_precision=False,
        apply_decay_param_fun=None,
    ):
        """Run single test combination.

        Args:
            ns_coeff: Newton-Schulz coefficient type.
            color_params: Optional list of param name substrings to assign
                custom color group (covers muon_sharding_optimizer L388-394).
            use_slice: If True, build split_concat_func_map for QKV/FFN params
                (covers muon.py L529, L535).
            explicit_dtype: If True, pass ns_matmul_dtype=paddle.float32 explicitly
                (covers muon.py L283).
            multi_precision: If True, enable FP32 master weights
                (covers muon.py L560-564, L574-575, L582-583).
            apply_decay_param_fun: Optional callable(param_name) -> bool
                (covers muon.py L443-446, L568-572).
        """
        # Allow env var to force multi_precision on
        if g_multi_precision:
            multi_precision = True
        weights = self._init_weights()

        # --- Distributed model (model_a) ---
        model_a = self._build_model(weights)

        # Assign custom color before optimizer construction
        if color_params:
            hcg = fleet.get_hybrid_communicate_group()
            sharding_group = hcg.get_sharding_parallel_group()
            for name, p in model_a.named_parameters():
                for pattern in color_params:
                    if pattern in name:
                        p.color = {
                            'color': 'test_color',
                            'group': sharding_group,
                        }
                        break

        split_concat_func_map = (
            self._build_split_concat_func_map(model_a) if use_slice else None
        )
        ns_dtype = paddle.float32 if explicit_dtype else None

        optimizer_a = self.build_optimizer(
            model_a,
            ns_coeff,
            split_concat_func_map=split_concat_func_map,
            ns_matmul_dtype=ns_dtype,
            multi_precision=multi_precision,
            apply_decay_param_fun=apply_decay_param_fun,
        )

        # --- Reference model (model_b, single-GPU) ---
        model_b = self._build_model(weights)

        split_concat_func_map_b = (
            self._build_split_concat_func_map(model_b) if use_slice else None
        )
        optimizer_b = self.build_optimizer(
            model_b,
            ns_coeff,
            split_concat_func_map=split_concat_func_map_b,
            ns_matmul_dtype=ns_dtype,
            multi_precision=multi_precision,
            apply_decay_param_fun=apply_decay_param_fun,
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

    def test_sharding_muon(self):
        """Test ns_coeff_type combinations + color/slice/dtype coverage.

        Phase 1: iterate all ns_coeff_types (basic, no slice, no color).
        Phase 2: custom color group + split_concat_func + explicit fp32 dtype.
        Phase 3: multi_precision=True (master weights for Muon 2D + AdamW 1D).
        Phase 4: apply_decay_param_fun that excludes some params from decay.
          Covers:
          - muon_sharding_optimizer.py L388-394: custom color from param.color dict
          - muon_sharding_optimizer.py L627-635, L665-667: fused gradient comm buffers
          - muon.py L283: explicit ns_matmul_dtype=paddle.float32
          - muon.py L443-446: apply_decay_param_fun with_decay=False (AdamW path)
          - muon.py L529: split_concat_func call
          - muon.py L535: MUON_DEBUG logging (via MUON_DEBUG=1 env)
          - muon.py L560-564, L574-575, L582-583: find_master=True (Muon path)
        """
        total = (
            len(NS_COEFF_TYPES) + 3
        )  # +1 color/slice/dtype, +1 multi_precision, +1 decay_fun
        passed = 0
        failed = []

        # Phase 1: ns_coeff_type combinations
        for ns_coeff in NS_COEFF_TYPES:
            print(f"\n[Muon Test] ns_coeff={ns_coeff}")
            try:
                self._run_single_test(ns_coeff)
                passed += 1
                print(f"[PASS] {ns_coeff}")
            except Exception as e:
                failed.append((ns_coeff, str(e)))
                print(f"[FAIL] {ns_coeff}: {e}")

        # Phase 2: color + split_concat_func + explicit fp32 dtype
        print("\n[Muon Test] color + split_concat_func + explicit fp32 dtype")
        try:
            self._run_single_test(
                "simple",
                color_params=["down_proj"],
                use_slice=True,
                explicit_dtype=True,
            )
            passed += 1
            print("[PASS] color + slice + dtype")
        except Exception as e:
            failed.append(("color+slice+dtype", str(e)))
            print(f"[FAIL] color + slice + dtype: {e}")

        # Phase 3: multi_precision=True — covers find_master branch in Muon
        # muon.py L560-564 (find_master=True), L574-575 (master_weight.scale_),
        # L582-583 (master_weight.subtract_ + assign)
        print("\n[Muon Test] multi_precision (master weights)")
        try:
            self._run_single_test(
                "simple",
                multi_precision=True,
            )
            passed += 1
            print("[PASS] multi_precision")
        except Exception as e:
            failed.append(("multi_precision", str(e)))
            print(f"[FAIL] multi_precision: {e}")

        # Phase 4: apply_decay_param_fun — covers with_decay=False branch
        # muon.py L443-446 (AdamW path: with_decay=False)
        # muon.py L568-572 (Muon path: with_decay=False)
        print("\n[Muon Test] apply_decay_param_fun (selective decay)")
        try:
            # Exclude all params from decay — exercises both AdamW and Muon with_decay=False
            self._run_single_test(
                "simple",
                apply_decay_param_fun=lambda name: False,
            )
            passed += 1
            print("[PASS] apply_decay_param_fun")
        except Exception as e:
            failed.append(("apply_decay_param_fun", str(e)))
            print(f"[FAIL] apply_decay_param_fun: {e}")

        print(f"\n{'=' * 60}")
        print(f"Muon Sharding Test Summary: {passed}/{total} passed")
        if failed:
            print("Failed combinations:")
            for ns, err in failed:
                print(f"  - {ns}: {err[:100]}...")
        print(f"{'=' * 60}")

        if failed:
            raise AssertionError(f"{len(failed)} test combinations failed")


if __name__ == "__main__":
    unittest.main()
