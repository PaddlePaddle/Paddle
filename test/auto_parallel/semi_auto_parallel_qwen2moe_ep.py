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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import nn
from paddle.io import BatchSampler, DataLoader


class Qwen2MoeConfig:
    model_type = "qwen2_moe"

    def __init__(
        self,
        norm_topk_prob=False,
        **kwargs,
    ):
        self.hidden_size = 8
        self.hidden_act = "silu"
        # MoE arguments
        self.moe_intermediate_size = 16
        self.shared_expert_intermediate_size = 32
        self.num_experts_per_tok = 2
        self.num_experts = 4
        self.norm_topk_prob = False
        self.tensor_parallel_degree = 2
        self.batch_num = 16
        self.batch_size = 8
        self.seq_len = 8
        self.hidden_dim = 8


class Qwen2MoeMLP(nn.Layer):
    def __init__(self, config: Qwen2MoeConfig, is_shared=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.moe_intermediate_size
            if not is_shared
            else config.shared_expert_intermediate_size
        )
        self.tensor_parallel_degree = config.tensor_parallel_degree

        if config.tensor_parallel_degree > 1:
            self.gate_proj = mpu.ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.up_proj = mpu.ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                gather_output=False,
                has_bias=False,
            )
            self.down_proj = mpu.RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                input_is_parallel=True,
                has_bias=False,
            )
        else:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias_attr=False
            )  # w1
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias_attr=False
            )  # w3
            self.down_proj = nn.Linear(
                self.intermediate_size, self.hidden_size, bias_attr=False
            )  # w2

        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MoeSparseMoEBlock(nn.Layer):
    def __init__(self, config: Qwen2MoeConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(
            config.hidden_size, self.num_experts, bias_attr=False
        )
        self.experts = nn.LayerList(
            [Qwen2MoeMLP(config) for _ in range(self.num_experts)]
        )

        self.shared_expert = Qwen2MoeMLP(config, is_shared=True)
        self.shared_expert_gate = nn.Linear(
            config.hidden_size, 1, bias_attr=False
        )

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])
        # router_logits: [batch_size * seq_len, num_experts]
        router_logits = self.gate(hidden_states)

        with paddle.amp.auto_cast(False):
            routing_weights = F.softmax(router_logits.astype("float32"), axis=1)
        routing_weights, selected_experts = paddle.topk(
            routing_weights, self.top_k, axis=-1
        )
        if (
            self.norm_topk_prob
        ):  # Note: Mixtral is set norm as default, Qwen2Moe is set to no norm
            routing_weights /= routing_weights.sum(axis=-1, keepdim=True)
        # we cast back to input dtype
        routing_weights = routing_weights.astype(hidden_states.dtype)

        final_hidden_states = paddle.zeros(
            [batch_size * seq_len, hidden_dim],
            dtype=hidden_states.dtype,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated.
        # shape: [num_experts, top_k, batch_size * seq_len]
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).transpose([2, 1, 0])

        # Loop over all available experts in the model and perform the computation on each expert.
        for expert_id in range(self.num_experts):
            expert_layer = self.experts[expert_id]
            idx, top_x = paddle.where(expert_mask[expert_id])

            if top_x.shape[0] == 0:
                continue

            current_state = paddle.gather(hidden_states, top_x.squeeze())
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx]
            )

            top_x = top_x.squeeze()
            if top_x.shape == []:
                top_x = paddle.to_tensor([top_x.item()])
            final_hidden_states = paddle.index_add_(
                final_hidden_states,
                top_x,
                0,
                current_hidden_states.astype(hidden_states.dtype),
            )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states))
            * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            [batch_size, seq_len, hidden_dim]
        )
        return final_hidden_states, router_logits


class RandomDataset(paddle.io.Dataset):
    def __init__(self, input_seq, output_seq, num_samples, return_dict=False):
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.num_samples = self.input_seq.shape[0]
        self.return_dict = return_dict

    def __getitem__(self, idx):
        if self.return_dict:
            return {
                "input_seq": self.input_seq[idx],
                "output_seq": self.output_seq[idx],
            }
        else:
            return self.input_seq[idx], self.output_seq[idx]

    def __len__(self):
        return self.num_samples


class TestQwen2moe:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        paddle.set_device(self._backend)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_optimizer(self, model, lr_scheduler=None):
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.01,
            parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
        )
        return optimizer

    def create_data_loader(self, config):
        nsamples = config.batch_size * config.batch_num
        input_seq = np.random.rand(
            nsamples, config.seq_len, config.hidden_size
        ).astype('float32')
        output_seq = np.random.rand(
            nsamples, config.seq_len, config.hidden_size
        ).astype('float32')
        train_dataset = RandomDataset(input_seq, output_seq, config.batch_size)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        return train_dataloader

    def build(self, config):
        strategy = dist.fleet.DistributedStrategy()
        strategy.tensor_parallel = True
        strategy.tensor_parallel_configs = {
            'tensor_parallel_degree': config.tensor_parallel_degree,
        }
        strategy.hybrid_configs = {
            "dp_degree": 2,
        }
        dist.fleet.init(is_collective=True, strategy=strategy)
        model = Qwen2MoeSparseMoEBlock(config)
        dataloader = self.create_data_loader(config)
        optimizer = self.create_optimizer(model)
        return model, dataloader, optimizer

    def train(self, config, model, train_dataloader, optimizer):
        tr_loss = float(0)
        global_step = 0
        model.train()
        losses = []
        for step, inputs in enumerate(train_dataloader()):
            inputs, labels = inputs
            logits = model(inputs)
            tr_loss = F.mse_loss(logits, labels, reduction='mean')
            tr_loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            losses.append(tr_loss.numpy())

        return losses

    def test_demo_net(self):
        self.set_seed(self._seed)
        config = Qwen2MoeConfig()
        config.run_ep = False
        model, train_dataloader, optimizer = self.build(config)

        loss = self.train(config, model, train_dataloader, optimizer)
        print("rp train success!!!")

    def run_test_case(self):
        self.test_demo_net()


if __name__ == "__main__":
    TestQwen2moe().run_test_case()
