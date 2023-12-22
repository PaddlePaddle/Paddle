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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import Replicate, Shard
from paddle.nn.functional.flash_attention import flash_attention

BATCH_NUM = 4
BATCH_SIZE = 16
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 1024 // 3 * 8
SEQ_LEN = 128
N_HEAD = 8


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(-0.1, 0.1)
    )


class LlamaAttention(nn.Layer):
    def __init__(self, param_prefix="", hidden_size=HIDDEN_SIZE, n_head=N_HEAD):
        super().__init__()
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")
        self.hidden_size = hidden_size
        self.num_heads = n_head
        self.head_dim = hidden_size // n_head
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, weight_attr_0)
        self.o_proj = nn.Linear(hidden_size, hidden_size, weight_attr_1)

    def forward(self, x):
        mix_layer = self.qkv_proj(x)
        target_shape = [0, 0, self.num_heads, 3 * self.head_dim]
        mix_layer = paddle.reshape(mix_layer, target_shape)
        mix_layer = paddle.cast(mix_layer, paddle.bfloat16)
        query_states, key_states, value_states = paddle.split(
            mix_layer, num_or_sections=3, axis=-1
        )
        attn_output, _ = flash_attention(
            query_states, key_states, value_states, causal=True
        )
        attn_output = paddle.cast(attn_output, paddle.float32)
        attn_output = attn_output.reshape(
            [BATCH_SIZE, SEQ_LEN, self.hidden_size]
        )
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMlp(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        bias_attr_0 = create_numpy_like_random(param_prefix + "_bias_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")
        bias_attr_1 = create_numpy_like_random(param_prefix + "_bias_1")
        weight_attr_2 = create_numpy_like_random(param_prefix + "_2")
        bias_attr_2 = create_numpy_like_random(param_prefix + "_bias_2")

        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, weight_attr_0, bias_attr_0
        )
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, weight_attr_1, bias_attr_1
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, weight_attr_2, bias_attr_2
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaRMSNorm(nn.Layer):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1.0

    def forward(self, hidden_states):
        with paddle.amp.auto_cast(False):
            variance = (
                hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            )
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )
        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight


class LlamaLayerDecoder(nn.Layer):
    def __init__(
        self,
        param_prefix="",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.self_attn = LlamaAttention(param_prefix + "_att", hidden_size)
        self.mlp = LlamaMlp(param_prefix + "_mlp")
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attn_layernorm = LlamaRMSNorm(hidden_size)

    def forward(self, x):
        residual = x
        hidden_states = self.input_layernorm(x)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TestLlamaDecoderForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype", "float32")
        self._backend = os.getenv("backend", "gpu")
        self._seed = eval(os.getenv("seed", "2023"))
        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def dp_mp_shard_fn(self, layer_name, layer, process_mesh):
        col_linear = ["qkv_proj", "gate_proj", "up_proj"]
        row_linear = ["o_proj", "down_proj"]

        def contains(a, b):
            return b in a

        is_col_linear = any(contains(layer_name, e) for e in col_linear)
        is_row_linear = any(contains(layer_name, e) for e in row_linear)

        if is_col_linear:
            layer.weight = dist.shard_tensor(
                layer.weight, process_mesh, [Replicate(), Shard(1)]
            )
            layer.bias = dist.shard_tensor(
                layer.bias, process_mesh, [Replicate(), Shard(0)]
            )

        if is_row_linear:
            layer.weight = dist.shard_tensor(
                layer.weight, process_mesh, [Replicate(), Shard(0)]
            )

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def init_input_data(self):
        input = np.random.random([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]).astype(
            self._dtype
        )
        input = paddle.to_tensor(input)
        return input

    def init_single_card_net_result(self):
        self.set_random_seed(self._seed)
        self.base_out, self.base_parameters = self.train_loop(
            LlamaLayerDecoder("demo_weight")
        )

    def train_loop(self, layer, process_mesh=None, shard_input=False):
        # run forward and backward

        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        for _ in range(5):
            input = self.init_input_data()
            if shard_input:
                input = dist.shard_tensor(input, process_mesh, shard_input)
            out = layer(input)
            loss = paddle.sum(out)
            loss.backward()
            opt.step()
            opt.clear_grad()
        return out, layer.parameters()

    def check_tensor_eq(self, a, b, rtol=1e-05, atol=0, verbose=True):
        if a is None:
            assert b is None
            return
        np1 = a.astype("float32").numpy()
        np2 = b.astype("float32").numpy()
        np.testing.assert_allclose(
            np1, np2, rtol=rtol, atol=atol, verbose=verbose
        )

    def check_placements(self, output, expected_placements):
        assert (
            output.placements == expected_placements
        ), f"{output.placements}  vs {expected_placements}"

    def get_shard_check_hook(self, dims_mapping, check_input=False):
        def check_func(layer, input, output=None):
            if check_input:
                if isinstance(input, tuple):
                    input = input[0]
                self.check_placements(input, dims_mapping)
            else:
                if isinstance(output, tuple):
                    output = output[0]
                self.check_placements(output, dims_mapping)

        return check_func

    # python -m paddle.distributed.launch --devices=0,1,2,3 semi_auto_parallel_for_llama_decoder_dp_mp.py
    def test_dp_mp(self):
        self.set_random_seed(self._seed)
        dp_mp_mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        dp_mp_layer = dist.shard_layer(
            LlamaLayerDecoder("mp_demo_weight"), dp_mp_mesh, self.dp_mp_shard_fn
        )
        input_layer_norm_post_hook = self.get_shard_check_hook(
            [dist.Shard(0), dist.Replicate()]
        )
        attn_pre_hook = self.get_shard_check_hook(
            [dist.Shard(0), dist.Replicate()], True
        )
        attn_post_hook = self.get_shard_check_hook(
            [dist.Shard(0), dist.Replicate()]
        )
        post_attn_layer_norm_pre_hook = self.get_shard_check_hook(
            [dist.Shard(0), dist.Replicate()], True
        )
        post_attn_layer_norm_post_hook = self.get_shard_check_hook(
            [dist.Shard(0), dist.Replicate()]
        )
        mlp_pre_hook = self.get_shard_check_hook(
            [dist.Shard(0), dist.Replicate()], True
        )
        mlp_post_hook = self.get_shard_check_hook(
            [dist.Shard(0), dist.Replicate()]
        )

        dp_mp_layer.input_layernorm.register_forward_post_hook(
            input_layer_norm_post_hook
        )

        dp_mp_layer.self_attn.register_forward_pre_hook(attn_pre_hook)
        dp_mp_layer.self_attn.register_forward_post_hook(attn_post_hook)

        dp_mp_layer.post_attn_layernorm.register_forward_pre_hook(
            post_attn_layer_norm_pre_hook
        )
        dp_mp_layer.post_attn_layernorm.register_forward_post_hook(
            post_attn_layer_norm_post_hook
        )

        dp_mp_layer.mlp.register_forward_pre_hook(mlp_pre_hook)
        dp_mp_layer.mlp.register_forward_post_hook(mlp_post_hook)

        dp_mp_out, dp_mp_parameters = self.train_loop(
            dp_mp_layer, dp_mp_mesh, shard_input=[Shard(0), Replicate()]
        )
        self.check_tensor_eq(dp_mp_out, self.base_out)
        for param, param_base in zip(dp_mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        if self._backend == "gpu":
            cuda_version_main = int(paddle.version.cuda().split(".")[0])
            device_prop_main = paddle.device.cuda.get_device_capability()[0]
            if cuda_version_main >= 11 and device_prop_main >= 8:
                self.test_dp_mp()


if __name__ == '__main__':
    TestLlamaDecoderForSemiAutoParallel().run_test_case()
