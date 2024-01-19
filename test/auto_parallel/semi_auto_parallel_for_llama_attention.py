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
from paddle import nn
from paddle.distributed import Replicate, Shard
from paddle.nn.functional.flash_attention import flash_attention

BATCH_NUM = 4
BATCH_SIZE = 16
HIDDEN_SIZE = 1024
SEQ_LEN = 128
N_HEAD = 8


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(0, 1)
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


class TestLlamaAttentionForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype", "float32")
        self._backend = os.getenv("backend", "gpu")
        self._seed = eval(os.getenv("seed", "2023"))

        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        paddle.set_device(self._backend)
        self.init_single_card_net_result()

    def mp_shard_fn(self, layer_name, layer, process_mesh):
        if layer_name == 'qkv_proj':
            layer.weight = dist.shard_tensor(
                layer.weight, process_mesh, [Shard(1)]
            )
            layer.bias = dist.shard_tensor(layer.bias, process_mesh, [Shard(0)])

        elif layer_name == 'o_proj':
            layer.weight = dist.shard_tensor(
                layer.weight, process_mesh, [Shard(0)]
            )

    def dp_mp_shard_fn(self, layer_name, layer, process_mesh):
        if layer_name == 'qkv_proj':
            layer.weight = dist.shard_tensor(
                layer.weight, process_mesh, [Replicate(), Shard(1)]
            )
            layer.bias = dist.shard_tensor(
                layer.bias, process_mesh, [Replicate(), Shard(0)]
            )

        elif layer_name == 'o_proj':
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
            LlamaAttention("demo_weight")
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

    def check_dim_mapping(self, output, expected_dim_mapping):
        assert (
            output.dist_attr.dims_mapping == expected_dim_mapping
        ), f"{output.dist_attr.dims_mapping}  vs {expected_dim_mapping}"

    def get_shard_check_hook(self, dims_mapping, check_input=False):
        def check_func(layer, input, output=None):
            if check_input:
                if isinstance(input, tuple):
                    input = input[0]
                self.check_dim_mapping(input, dims_mapping)
            else:
                if isinstance(output, tuple):
                    output = output[0]
                self.check_dim_mapping(output, dims_mapping)

        return check_func

    def test_dp(self):
        self.set_random_seed(self._seed)

        dp_layer = LlamaAttention("dp_demo_weight")
        qkv_proj_pre_hook = self.get_shard_check_hook([0, -1, -1], True)
        qkv_proj_post_hook = self.get_shard_check_hook([0, -1, -1])
        o_proj_pre_hook = self.get_shard_check_hook([0, -1, -1], True)
        o_proj_post_hook = self.get_shard_check_hook([0, -1, -1])

        dp_layer.qkv_proj.register_forward_pre_hook(qkv_proj_pre_hook)
        dp_layer.qkv_proj.register_forward_post_hook(qkv_proj_post_hook)
        dp_layer.o_proj.register_forward_pre_hook(o_proj_pre_hook)
        dp_layer.o_proj.register_forward_post_hook(o_proj_post_hook)

        dp_out, dp_parameters = self.train_loop(
            dp_layer,
            self._mesh,
            shard_input=[Shard(0)],
        )
        self.check_tensor_eq(dp_out, self.base_out)
        for param, param_base in zip(dp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def test_mp(self):
        self.set_random_seed(self._seed)

        mp_layer = dist.shard_layer(
            LlamaAttention("mp_demo_weight"), self._mesh, self.mp_shard_fn
        )

        qkv_proj_post_hook = self.get_shard_check_hook([-1, -1, 0])
        o_proj_pre_hook = self.get_shard_check_hook([-1, -1, 0], True)
        o_proj_post_hook = self.get_shard_check_hook([-1, -1, -1])

        mp_layer.qkv_proj.register_forward_post_hook(qkv_proj_post_hook)
        mp_layer.o_proj.register_forward_pre_hook(o_proj_pre_hook)
        mp_layer.o_proj.register_forward_post_hook(o_proj_post_hook)

        mp_out, mp_parameters = self.train_loop(mp_layer, self._mesh)
        self.check_tensor_eq(mp_out, self.base_out)
        for param, param_base in zip(mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    # python -m paddle.distributed.launch --devices=0,1,2,3 semi_auto_parallel_for_llama_attention.py
    def test_dp_mp(self):
        self.set_random_seed(self._seed)
        dp_mp_mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        dp_mp_layer = dist.shard_layer(
            LlamaAttention("mp_demo_weight"), dp_mp_mesh, self.dp_mp_shard_fn
        )

        qkv_proj_post_hook = self.get_shard_check_hook([0, -1, 1])
        o_proj_pre_hook = self.get_shard_check_hook([0, -1, 1], True)
        o_proj_post_hook = self.get_shard_check_hook([0, -1, -1])

        dp_mp_layer.qkv_proj.register_forward_post_hook(qkv_proj_post_hook)
        dp_mp_layer.o_proj.register_forward_pre_hook(o_proj_pre_hook)
        dp_mp_layer.o_proj.register_forward_post_hook(o_proj_post_hook)

        dp_mp_out, dp_mp_parameters = self.train_loop(
            dp_mp_layer, dp_mp_mesh, shard_input=[Shard(0), Replicate()]
        )
        self.check_tensor_eq(dp_mp_out, self.base_out)
        for param, param_base in zip(dp_mp_parameters, self.base_parameters):
            self.check_tensor_eq(param, param_base)
            self.check_tensor_eq(param.grad, param_base.grad)

    def run_test_case(self):
        self.test_dp()
        self.test_mp()
        # self.test_dp_mp()


if __name__ == '__main__':
    TestLlamaAttentionForSemiAutoParallel().run_test_case()
