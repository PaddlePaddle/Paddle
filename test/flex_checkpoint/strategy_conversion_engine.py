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

# strategy_conversion_engine.py
import argparse
import hashlib

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
)

# ==============================================================================
# 1. Model Definitions
# A model zoo with simple models supporting different parallelism strategies.
# ==============================================================================


class MLPBlock(nn.Layer):
    """
    A basic building block compatible with Tensor Parallelism,
    mimicking a transformer's FFN layer.
    """

    def __init__(self, hidden_size=32):
        super().__init__()
        self.linear1 = ColumnParallelLinear(
            hidden_size, hidden_size * 4, has_bias=True, gather_output=False
        )
        self.relu = nn.ReLU()
        self.linear2 = RowParallelLinear(
            hidden_size * 4, hidden_size, has_bias=True, input_is_parallel=True
        )

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class UnifiedMLP(nn.Sequential):
    """
    A unified model composed of multiple MLPBlocks.
    This sequential structure is suitable for all parallelism types:
    - TP is handled inside each MLPBlock.
    - PP wraps this entire Sequential model.
    - DP/EP treats this entire Sequential model as a single unit.
    """

    def __init__(self, hidden_size=32, num_blocks=4):
        super().__init__(*[MLPBlock(hidden_size) for _ in range(num_blocks)])


class Top1Router(nn.Layer):
    """A simple Top-1 Gating network for MoE."""

    def __init__(self, d_model, num_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        expert_weights, expert_indices = paddle.topk(gate_logits, k=1, axis=-1)
        return nn.functional.softmax(expert_weights, axis=-1), expert_indices


class MoELayer(nn.Layer):
    """
    A more robust MoE layer that handles both EP > 1 (distributed)
    and EP = 1 (local) scenarios.
    """

    def __init__(self, d_model, num_experts, num_blocks=2, moe_group=None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.moe_group = moe_group
        self.ep_world_size = moe_group.nranks if moe_group else 1

        self.router = Top1Router(d_model, num_experts)
        self.experts = nn.LayerList(
            [UnifiedMLP(d_model, num_blocks) for _ in range(self.num_experts)]
        )

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape([-1, self.d_model])
        expert_weights, expert_indices = self.router(x)
        final_output = paddle.zeros_like(x)

        if self.ep_world_size > 1:
            # Simplified distributed routing for testing purposes.
            ep_rank = dist.get_rank(self.moe_group)
            for i in range(self.num_experts):
                if i % self.ep_world_size == ep_rank:
                    mask = (expert_indices == i).astype('float32')
                    expert_output = self.experts[i](x)
                    final_output += expert_output * mask
        else:
            # Local routing for EP = 1
            for i in range(self.num_experts):
                token_mask = (expert_indices == i).squeeze(-1)
                if not token_mask.any():
                    continue
                selected_tokens = x[token_mask]
                selected_weights = expert_weights[token_mask]
                expert_output = self.experts[i](selected_tokens)
                indices_to_scatter = paddle.where(token_mask)[0]
                final_output = paddle.scatter(
                    final_output,
                    indices_to_scatter,
                    expert_output * selected_weights,
                    overwrite=False,
                )

        return final_output.reshape(original_shape)


# ==============================================================================
# 2. Core Logic (Environment Setup, Execution, and Verification)
# ==============================================================================


def get_model_and_strategy(args, hcg):
    """Builds model and DistributedStrategy based on parsed arguments."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp,
        "mp_degree": args.tp,
        "pp_degree": args.pp,
    }

    if args.model_type == "moe":
        model = MoELayer(d_model=32, num_experts=4)
    else:
        model = UnifiedMLP()

    if args.ep > 1:
        model = MoELayer(
            d_model=32, num_experts=4, moe_group=hcg.get_data_parallel_group()
        )
        strategy.hybrid_configs["ep_degree"] = args.ep
    elif args.pp > 1:
        # For PP, the model must be wrapped by PipelineLayer
        model = fleet.meta_parallel.PipelineLayer(
            layers=model, num_stages=args.pp, topology=hcg.topology()
        )

    return model, strategy


def setup_execution_environment(config_args):
    """A unified function to initialize Fleet and the model."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": config_args.dp,
        "mp_degree": config_args.tp,
        "pp_degree": config_args.pp,
    }

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()

    model, strategy = get_model_and_strategy(config_args, hcg)

    # Re-initialize with the final strategy (in case ep_degree was added)
    fleet.init(is_collective=True, strategy=strategy)

    return model


def verify_by_md5(sd1, sd2):
    """Compares two state_dicts by the MD5 hash of each parameter."""

    def get_tensor_md5(tensor):
        return hashlib.md5(tensor.numpy().tobytes()).hexdigest()

    assert sd1.keys() == sd2.keys(), (
        f"State dicts have different keys! Got {sd1.keys()} vs {sd2.keys()}"
    )
    for key in sd1.keys():
        md5_1 = get_tensor_md5(sd1[key])
        md5_2 = get_tensor_md5(sd2[key])
        assert md5_1 == md5_2, (
            f"MD5 mismatch for param '{key}': baseline={md5_1} vs roundtrip={md5_2}"
        )


def run_step1_save_source(args):
    """Step 1: In the source configuration, save a distributed checkpoint."""
    model = setup_execution_environment(args.src)
    dist.save_state_dict(model.sharded_state_dict(), args.src_ckpt_path)


def run_step2_convert(args):
    """Step 2: In the target configuration, load the source checkpoint and resave."""
    model = setup_execution_environment(args.tgt)
    dist.load_state_dict(model.sharded_state_dict(), args.src_ckpt_path)
    dist.save_state_dict(model.sharded_state_dict(), args.tgt_ckpt_path)


def run_step3_verify(args):
    """Step 3: In the source configuration, load both checkpoints and compare them."""
    # 1. Create the "round-trip" model by loading the target checkpoint
    model_roundtrip = setup_execution_environment(args.src)
    dist.load_state_dict(
        model_roundtrip.sharded_state_dict(), args.tgt_ckpt_path
    )

    # 2. Create the "baseline" model by loading the original source checkpoint
    model_baseline = setup_execution_environment(args.src)
    dist.load_state_dict(
        model_baseline.sharded_state_dict(), args.src_ckpt_path
    )

    dist.barrier()

    # 3. Each rank verifies its own part of the state_dict.
    # This works for all strategies, including Pipeline Parallelism.
    final_sd = model_roundtrip.state_dict()
    initial_sd = model_baseline.state_dict()

    if final_sd and initial_sd:
        verify_by_md5(initial_sd, final_sd)


# ==============================================================================
# 3. Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["save_source", "convert", "verify"],
    )
    parser.add_argument("--src_ckpt_path", type=str)
    parser.add_argument("--tgt_ckpt_path", type=str)
    parser.add_argument(
        "--model_type",
        default="mlp",
        choices=["mlp", "moe"],
        help="Model architecture.",
    )

    # Add all strategy parameters dynamically for source and target
    for prefix in ["src", "tgt"]:
        for p in ["world_size", "tp", "dp", "pp", "ep"]:
            parser.add_argument(f"--{prefix}_{p}", type=int, default=0)

    args = parser.parse_args()

    # Reorganize parsed args into src/tgt namespaces
    def organize_args(prefix):
        config = {
            p: getattr(args, f"{prefix}_{p}")
            for p in ["world_size", "tp", "dp", "pp", "ep"]
        }
        config["model_type"] = args.model_type
        # Default parallelism degree to 1 if not specified
        if config["tp"] == 0:
            config["tp"] = 1
        if config["dp"] == 0:
            config["dp"] = 1
        if config["pp"] == 0:
            config["pp"] = 1
        if config["ep"] == 0:
            config["ep"] = 1
        return argparse.Namespace(**config)

    args.src = organize_args("src")
    args.tgt = organize_args("tgt")

    # Execute the requested step
    engine = {
        "save_source": run_step1_save_source,
        "convert": run_step2_convert,
        "verify": run_step3_verify,
    }
    engine[args.step](args)
