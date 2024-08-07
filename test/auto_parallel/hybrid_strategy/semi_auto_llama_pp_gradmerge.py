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
from functools import reduce

import numpy as np
from semi_auto_parallel_llama_model import (
    LlamaForCausalLMAuto,
    LlamaPretrainingCriterionAuto,
    get_mesh,
)

import paddle
import paddle.distributed as dist
from paddle import LazyGuard
from paddle.io import BatchSampler, DataLoader, Dataset


class Config:
    vocab_size = 32000
    hidden_size = 4096
    intermediate_size = 11008
    max_position_embeddings = 2048
    seq_length = 2048
    num_hidden_layers = 4
    num_attention_heads = 32
    num_key_value_heads = 32
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    use_cache = True
    use_flash_attention = False
    sequence_parallel = False
    rope = True
    recompute = False
    recompute_granularity = None
    use_lazy_init = False
    virtual_pp_degree = 1


class RandomDataset(Dataset):
    def __init__(self, seq_len, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len]).astype("int64")
        label = (np.random.uniform(size=[self.seq_len]) * 10).astype("int64")
        return input, label

    def __len__(self):
        return self.num_samples


def create_optimizer(model, lr_scheduler):
    decay_parameters = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    def apply_decay_param_fun(x):
        return x in decay_parameters

    # test global_clip in auto_parallel
    if os.getenv("use_param_group") == "true":
        param_group = {}
        param_group["params"] = list(model.parameters())
        param_group["weight_decay"] = 0.01
        param_group["grad_clip"] = paddle.nn.ClipGradByGlobalNorm(1.0)
        optimizer = paddle.optimizer.adamw.AdamW(
            learning_rate=lr_scheduler,
            apply_decay_param_fun=apply_decay_param_fun,
            parameters=[param_group],
        )
    else:
        optimizer = paddle.optimizer.adamw.AdamW(
            learning_rate=lr_scheduler,
            apply_decay_param_fun=apply_decay_param_fun,
            parameters=model.parameters(),
            weight_decay=0.01,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
        )
    return optimizer


class TestLlamaAuto:
    def __init__(self):
        self.config = Config()
        self.dp = int(os.getenv("dp"))
        self.mp = int(os.getenv("mp"))
        self.pp = int(os.getenv("pp"))
        if os.getenv("virtual_pp_degree"):
            self.config.virtual_pp_degree = int(os.getenv("virtual_pp_degree"))
        if os.getenv("use_sp") == "true":
            self.config.sequence_parallel = True
        if os.getenv("recompute") == "true":
            self.config.recompute = True
        self.config.recompute_granularity = os.getenv("recompute_granularity")
        self.gradient_accumulation_steps = int(os.getenv("acc_step"))
        self.only_static = os.getenv("only_static")

        if self.config.virtual_pp_degree == 1:
            self.schedule_mode = "1F1B"
        elif self.config.virtual_pp_degree > 1:
            self.schedule_mode = "VPP"

        self.init_dist_env()

    def init_dist_env(self):
        order = ["dp", "pp", "mp"]
        dp_degree = self.dp
        mp_degree = self.mp
        pp_degree = self.pp
        degree = [dp_degree, pp_degree, mp_degree]
        mesh_dims = list(filter(lambda x: x[1] > 1, list(zip(order, degree))))
        if not mesh_dims:
            mesh_dims = [("dp", 1)]
        dim_names = [mesh_dim[0] for mesh_dim in mesh_dims]
        mesh_shape = [mesh_dim[1] for mesh_dim in mesh_dims]
        mesh_arr = np.arange(
            0, reduce(lambda x, y: x * y, mesh_shape, 1)
        ).reshape(mesh_shape)
        global_mesh = dist.ProcessMesh(mesh_arr, dim_names)
        dist.auto_parallel.set_mesh(global_mesh)

    def run_llama(self, to_static=0):
        if self.only_static and to_static == 0:
            return

        if self.config.use_lazy_init:
            with LazyGuard():
                model = LlamaForCausalLMAuto(self.config)
            for param in model.parameters():
                assert not param._is_initialized()
                param.initialize()
        else:
            model = LlamaForCausalLMAuto(self.config)
        model = LlamaForCausalLMAuto(self.config)
        criterion = LlamaPretrainingCriterionAuto(self.config)

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)
        optimizer = dist.shard_optimizer(optimizer)

        micro_bsz = 2
        global_bsz = micro_bsz * self.dp * self.gradient_accumulation_steps
        run_step = 5
        total_sample_num = run_step * global_bsz
        global_step = 1
        tr_loss = float(0)

        if not to_static:
            train_dataset = RandomDataset(
                self.config.seq_length, total_sample_num
            )
            train_sampler = BatchSampler(
                train_dataset,
                batch_size=micro_bsz,
                shuffle=True,
                drop_last=True,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=0,
            )

            model.train()
            for epoch_idx in range(1):
                for step, inputs in enumerate(train_dataloader):
                    input_ids, labels = inputs
                    logits = model(input_ids)
                    tr_loss_step = criterion(logits, labels)

                    if self.gradient_accumulation_steps > 1:
                        tr_loss_step /= self.gradient_accumulation_steps

                    tr_loss_step.backward()
                    tr_loss += tr_loss_step

                    if global_step % self.gradient_accumulation_steps == 0:
                        print(
                            f"step: {global_step // self.gradient_accumulation_steps}  loss: {tr_loss.numpy()}"
                        )
                        optimizer.step()
                        optimizer.clear_grad()
                        lr_scheduler.step()
                        tr_loss = 0

                    global_step += 1
                    if global_step // self.gradient_accumulation_steps >= 10:
                        break
        else:
            strategy = dist.Strategy()
            if self.pp > 1 and self.gradient_accumulation_steps > 1:
                strategy.pipeline.enable = True
                strategy.pipeline.accumulate_steps = (
                    self.gradient_accumulation_steps
                )
                strategy.pipeline.micro_batch_size = micro_bsz
                strategy.pipeline.schedule_mode = self.schedule_mode
                strategy.pipeline.vpp_degree = self.config.virtual_pp_degree
                strategy.pipeline.vpp_seg_method = "LlamaDecoderLayerAuto"
            elif self.gradient_accumulation_steps > 1:
                strategy.gradient_merge.enable = True
                strategy.gradient_merge.k_steps = (
                    self.gradient_accumulation_steps
                )
                strategy.gradient_merge.avg = True

            train_dataset = RandomDataset(
                self.config.seq_length, total_sample_num
            )
            train_sampler = BatchSampler(
                train_dataset,
                batch_size=global_bsz,
                shuffle=True,
                drop_last=True,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=0,
            )

            dist_loader = dist.shard_dataloader(
                train_dataloader,
                meshes=[get_mesh(0), get_mesh(-1)],
                shard_dims="dp",
            )
            dist_model = dist.to_static(
                model, dist_loader, criterion, optimizer, strategy=strategy
            )

            def validate_batch(batch):
                if self.gradient_accumulation_steps == 1 or self.pp > 1:
                    batches = [batch]
                else:
                    split_batches = [
                        np.split(
                            np.array(b), self.gradient_accumulation_steps, 0
                        )
                        for b in batch
                    ]
                    batches = []
                    for i in range(len(split_batches[0])):
                        micro_batch = [
                            split_batch[i] for split_batch in split_batches
                        ]
                        batches.append(micro_batch)
                return batches

            dist_model.train()
            for epoch_idx in range(1):
                for step, inputs in enumerate(dist_loader()):
                    batches = validate_batch(inputs)
                    for micro_batch in batches:
                        input_ids, labels = micro_batch
                        tr_loss_step = dist_model(input_ids, labels)

                        if (
                            tr_loss_step is not None
                            and self.gradient_accumulation_steps > 1
                        ):
                            tr_loss_step = np.sum(tr_loss_step)
                            tr_loss_step /= self.gradient_accumulation_steps

                        if tr_loss_step:
                            tr_loss += tr_loss_step

                    print(f"step: {step}  loss: {np.array(tr_loss)}")
                    lr_scheduler.step()
                    tr_loss = float(0)

                    if step >= run_step:
                        break

    def run_test_cases(self):
        self.run_llama(to_static=0)
        self.run_llama(to_static=1)


if __name__ == '__main__':
    TestLlamaAuto().run_test_cases()
