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

import hashlib
import os
import random
from functools import reduce

import numpy as np
from semi_auto_parallel_llama_model import (
    LlamaForCausalLMAuto,
    LlamaPretrainingCriterionAuto,
    get_mesh,
)

import paddle
import paddle.distributed as dist
from paddle.io import BatchSampler, DataLoader, Dataset


class Config:
    vocab_size = 320
    hidden_size = 8
    intermediate_size = 64
    max_position_embeddings = 8
    seq_length = 8

    num_hidden_layers = 2
    num_attention_heads = 2
    num_key_value_heads = 2
    initializer_range = 0.02
    rms_norm_eps = 1e-6
    use_cache = True
    use_flash_attention = False
    sequence_parallel = False
    rope = True


class RandomDataset(Dataset):
    def __init__(self, seq_len, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.full([self.seq_len], index, dtype="int64")
        label = np.array([index] * 8)

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
        if os.getenv("use_sp") == "true":
            self.config.sequence_parallel = True
        self.gradient_accumulation_steps = int(os.getenv("acc_step"))
        self.config.recompute = False
        self.config.sep_parallel_degree = 1

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
        paddle.seed(1024)
        np.random.seed(1024)
        random.seed(1024)

    def run_dynamic(self):
        model = LlamaForCausalLMAuto(self.config)
        criterion = LlamaPretrainingCriterionAuto(self.config)

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)

        train_dataset = RandomDataset(self.config.seq_length)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=2,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=[get_mesh(0), get_mesh(1)],
            shard_dims="dp",
        )

        tr_loss = float(0)
        model.train()
        #####
        for step, inputs in enumerate(dist_loader()):
            input_ids, labels = inputs
            logits = model(input_ids)
            tr_loss_step = criterion(logits, labels)
            if self.gradient_accumulation_steps > 1:
                tr_loss_step /= self.gradient_accumulation_steps

            tr_loss_step.backward()
            tr_loss += tr_loss_step

            if step % self.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.clear_grad()
                lr_scheduler.step()
                tr_loss = 0

            if step >= 9:
                break
            if int(dist.get_rank()) in [2, 3, 6, 7]:
                assert tr_loss_step._is_initialized()
            else:
                assert not tr_loss_step._is_initialized()

        return tr_loss_step._md5sum()

    def run_dy2static(self):
        model = LlamaForCausalLMAuto(self.config)
        criterion = LlamaPretrainingCriterionAuto(self.config)

        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)

        train_dataset = RandomDataset(self.config.seq_length)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=2,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
        )
        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=[get_mesh(0), get_mesh(1)],
            shard_dims="dp",
        )

        strategy = None
        if self.gradient_accumulation_steps > 1:
            strategy = dist.Strategy()
            strategy.pipeline.accumulate_steps = (
                self.gradient_accumulation_steps
            )

        dist_model = dist.to_static(
            model, dist_loader, criterion, optimizer, strategy=strategy
        )

        dist_model.train()
        loss = None
        for step, inputs in enumerate(dist_loader()):
            input_ids, labels = inputs
            loss = dist_model(input_ids, labels)
            lr_scheduler.step()
            if step >= 9:
                break
            if int(dist.get_rank()) in [2, 3, 6, 7]:
                assert loss is not None
            else:
                assert loss is None

        numpy_array = np.array(loss)
        array_bytes = numpy_array.tobytes()
        loss_md5 = hashlib.md5(array_bytes).hexdigest()
        return loss_md5

    def run_test_cases(self):
        self.init_dist_env()
        dy_loss_md5 = self.run_dynamic()
        self.init_dist_env()
        st_loss_md5 = self.run_dy2static()
        if int(dist.get_rank()) in [2, 3, 6, 7]:
            assert dy_loss_md5 == st_loss_md5


if __name__ == '__main__':
    TestLlamaAuto().run_test_cases()
