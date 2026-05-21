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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.fsdp.fully_shard import fully_shard
from paddle.io import DataLoader, Dataset, DistributedBatchSampler


class RandomDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10):
        self.num_samples = num_samples
        self.input_dim = input_dim

    def __getitem__(self, idx):
        np.random.seed(2025)
        data = np.random.randn(self.input_dim).astype('float32')
        label = np.random.randn(self.input_dim).astype('float32')
        return data, label

    def __len__(self):
        return self.num_samples


class StandardMLPExpert(paddle.nn.Layer):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = paddle.nn.Linear(d_model, d_hidden)
        self.fc2 = paddle.nn.Linear(d_hidden, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)
        return x


class TransformerLayer(paddle.nn.Layer):
    def __init__(self, d_model, d_hidden, num_experts=2, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        self.gate = paddle.nn.Linear(d_model, num_experts, bias_attr=False)

        self.experts = paddle.nn.LayerList(
            [StandardMLPExpert(d_model, d_hidden) for _ in range(num_experts)]
        )

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape([-1, self.d_model])

        gate_logits = self.gate(x)
        gate_probs = paddle.nn.functional.softmax(gate_logits, axis=-1)

        topk_probs, topk_indices = paddle.topk(gate_probs, self.top_k, axis=-1)

        output = paddle.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]
            expert_weight = topk_probs[:, k].unsqueeze(-1)
            for i in range(self.num_experts):
                mask = (expert_idx == i).astype(x.dtype).unsqueeze(-1)
                expert_out = self.experts[i](x)
                output = output + mask * expert_weight * expert_out

        output = output.reshape(orig_shape)
        return output


class SimpleMoEModel(paddle.nn.Layer):
    def __init__(self, d_model=10, d_hidden=20, num_experts=2, top_k=1):
        super().__init__()
        self.config = type('Config', (), {'num_experts_per_tok': top_k})()
        self.embed = paddle.nn.Linear(d_model, d_model)
        self.moe = TransformerLayer(d_model, d_hidden, num_experts, top_k)
        self.output = paddle.nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = self.moe(x)
        x = x.squeeze(1)
        x = self.output(x)
        return x


class TestSemiAutoParallelFSDP:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["dp"])
        dist.auto_parallel.set_mesh(self._mesh)
        self.gradient_accumulation_steps = 2

    def create_dist_loader(self, batch_size):
        dataset = RandomDataset(num_samples=10, input_dim=10)

        sampler = DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        loader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
        )

        return loader

    def test_sharding_stage_3(self):
        paddle.seed(self._seed)
        model = paddle.nn.Linear(10, 10)
        data_loader = self.create_dist_loader(1)
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self._mesh],
            shard_dims="dp",
        )
        opt = paddle.optimizer.AdamW(parameters=model.parameters())

        # use sharding stage 3
        opt = dist.shard_optimizer(opt, dist.ShardingStage3("dp", self._mesh))

        stage_losses = []
        tr_loss_add = float(0)
        step = 0
        for batch in dist_loader:
            tr_loss = model(batch[0])
            tr_loss.backward()
            tr_loss_add += tr_loss
            if (step + 1) % self.gradient_accumulation_steps == 0:
                tr_loss_add /= self.gradient_accumulation_steps
                tr_loss = tr_loss_add
                opt.step()
                opt.clear_grad()
            step += 1
            stage_losses.append(tr_loss._local_value()._md5sum())
        return stage_losses

    def test_fsdp(self):
        paddle.seed(self._seed)
        model = paddle.nn.Linear(10, 10)
        data_loader = self.create_dist_loader(1)
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self._mesh],
            shard_dims="dp",
        )
        opt = paddle.optimizer.AdamW(parameters=model.parameters())

        # use FSDP
        model = fully_shard(
            model, mesh=self._mesh, enable_tensor_fusion_and_overlap=False
        )

        stage_losses = []
        tr_loss_add = float(0)
        step = 0
        for batch in dist_loader:
            tr_loss = model(batch[0])
            tr_loss.backward()
            tr_loss_add += tr_loss
            if (step + 1) % self.gradient_accumulation_steps == 0:
                tr_loss_add /= self.gradient_accumulation_steps
                tr_loss = tr_loss_add
                opt.step()
                opt.clear_grad()
            step += 1
            stage_losses.append(tr_loss._local_value()._md5sum())
        return stage_losses

    def test_fsdp_with_auto_dp(self):
        dist.enable_auto_dp()
        paddle.seed(self._seed)
        model = paddle.nn.Linear(10, 10)
        data_loader = self.create_dist_loader(2)
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self._mesh],
        )
        opt = paddle.optimizer.AdamW(parameters=model.parameters())

        # use FSDP
        model = fully_shard(
            model, mesh=self._mesh, enable_tensor_fusion_and_overlap=False
        )

        stage_losses = []
        tr_loss_add = float(0)
        step = 0
        for batch in dist_loader:
            tr_loss = model(batch[0])
            tr_loss.backward()
            tr_loss_add += tr_loss
            if (step + 1) % self.gradient_accumulation_steps == 0:
                tr_loss_add /= self.gradient_accumulation_steps
                tr_loss = tr_loss_add
                opt.step()
                opt.clear_grad()
            step += 1
            stage_losses.append(tr_loss._local_value()._md5sum())
        return stage_losses

    def test_fsdp_with_tensor_fusion_and_overlap(self):
        dist.init_parallel_env()
        dist.enable_auto_dp()
        paddle.seed(self._seed)
        model = paddle.nn.Linear(10, 10)
        data_loader = self.create_dist_loader(2)
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self._mesh],
        )
        opt = paddle.optimizer.AdamW(parameters=model.parameters())

        # use FSDP with tensor_fusion and overlap
        model = fully_shard(
            model, mesh=self._mesh, enable_tensor_fusion_and_overlap=True
        )

        stage_losses = []
        tr_loss_add = float(0)
        step = 0
        for batch in dist_loader:
            tr_loss = model(batch[0])
            tr_loss.backward()
            tr_loss_add += tr_loss
            if (step + 1) % self.gradient_accumulation_steps == 0:
                tr_loss_add /= self.gradient_accumulation_steps
                tr_loss = tr_loss_add
                opt.step()
                opt.clear_grad()
            step += 1
            stage_losses.append(tr_loss._local_value()._md5sum())
        return stage_losses

    def test_fsdp_with_moe(self):
        dist.init_parallel_env()
        dist.enable_auto_dp()
        paddle.seed(self._seed)

        model = SimpleMoEModel(d_model=10, d_hidden=20, num_experts=2, top_k=1)

        data_loader = self.create_dist_loader(2)
        dist_loader = dist.shard_dataloader(
            dataloader=data_loader,
            meshes=[self._mesh],
        )
        opt = paddle.optimizer.AdamW(parameters=model.parameters())

        model = fully_shard(
            model, mesh=self._mesh, enable_tensor_fusion_and_overlap=True
        )

        stage_losses = []
        tr_loss_add = float(0)
        step = 0
        for batch in dist_loader:
            tr_loss = model(batch[0])
            tr_loss.backward()
            tr_loss_add += tr_loss
            if (step + 1) % self.gradient_accumulation_steps == 0:
                tr_loss_add /= self.gradient_accumulation_steps
                tr_loss = tr_loss_add
                opt.step()
                opt.clear_grad()
            step += 1
            stage_losses.append(tr_loss._local_value()._md5sum())
        return stage_losses

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        elif self._backend == "gpu":
            paddle.set_device("gpu:" + str(dist.get_rank()))
        else:
            raise ValueError("Only support cpu or gpu backend.")

        losses_stage_3 = self.test_sharding_stage_3()
        losses_fsdp = self.test_fsdp()
        assert losses_stage_3 == losses_fsdp

        losses_fsdp_with_auto_dp = self.test_fsdp_with_auto_dp()
        assert losses_fsdp_with_auto_dp == losses_fsdp

        losses_fsdp_fusion = self.test_fsdp_with_tensor_fusion_and_overlap()
        assert losses_fsdp_fusion == losses_fsdp

        losses_fsdp_moe = self.test_fsdp_with_moe()
        assert len(losses_fsdp_moe) > 0, "MoE test should produce losses"


if __name__ == '__main__':
    TestSemiAutoParallelFSDP().run_test_case()
