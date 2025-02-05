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

os.environ["FLAGS_enable_pir_api"] = str(1)
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

        self.strategy = dist.Strategy()

        # amp config
        amp = self.strategy._amp
        if os.getenv("amp"):
            amp.enable = os.getenv("amp")
        if os.getenv("amp_dtype"):
            amp.dtype = os.getenv("amp_dtype")
        if os.getenv("amp_level"):
            amp.level = os.getenv("amp_level")
        if os.getenv("amp_master_grad"):
            amp.use_master_grad = os.getenv("amp_master_grad")
        if os.getenv("scale_loss"):
            amp.init_loss_scaling = os.getenv("scale_loss")
        if os.getenv("amp_custom_black_list"):
            amp.custom_black_list = os.getenv("amp_custom_black_list")
        if os.getenv("amp_custom_white_list"):
            amp.custom_white_list = os.getenv("amp_custom_white_list")

        self.gradient_accumulation_steps = 1
        if os.getenv("acc_step"):
            self.gradient_accumulation_steps = int(os.getenv("acc_step"))

        if self.gradient_accumulation_steps > 1:
            self.strategy.gradient_merge.enable = True
            self.strategy.gradient_merge.k_steps = (
                self.gradient_accumulation_steps
            )
            self.strategy.gradient_merge.avg = False

        self.config.recompute = False
        self.config.sep_parallel_degree = 1

        self.run_step = 10
        self.run_step_dy2static = (
            self.run_step // self.gradient_accumulation_steps
        )

    def run_llama(self, to_static=0):
        # model
        model = LlamaForCausalLMAuto(self.config)
        criterion = LlamaPretrainingCriterionAuto(self.config)
        if self.strategy._amp.enable and self.strategy._amp.level == "O2":
            paddle.amp.decorate(
                models=model,
                level=self.strategy._amp.level,
                dtype=self.strategy._amp.dtype,
                master_grad=self.strategy._amp.use_master_grad,
            )

        # optimizer
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=0.0001, warmup_steps=2, start_lr=0, end_lr=0.0001
        )
        optimizer = create_optimizer(model, lr_scheduler)
        optimizer = dist.shard_optimizer(optimizer)

        # dataloader
        train_dataset = RandomDataset(self.config.seq_length)
        train_sampler = BatchSampler(
            train_dataset,
            batch_size=2,
            shuffle=True,
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

        if to_static:
            model = dist.to_static(
                model, dist_loader, criterion, optimizer, strategy=self.strategy
            )
        model.train()
        losses = []
        for step, inputs in enumerate(dist_loader()):
            if step >= self.run_step:
                break
            input_ids, labels = inputs
            if to_static:
                loss = model(input_ids, labels)
                if loss is None:
                    numpy_array = np.array([])
                else:
                    numpy_array = np.array(loss)
                losses.append(numpy_array)
                array_bytes = numpy_array.tobytes()
            else:
                logits = model(input_ids)
                loss = criterion(logits, labels)
                losses.append(np.array(loss))
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
            lr_scheduler.step()
        return losses

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
        tr_loss_add = float(0)
        model.train()
        #####
        for step, inputs in enumerate(dist_loader()):
            if step >= self.run_step:
                break

            input_ids, labels = inputs
            logits = model(input_ids)
            tr_loss_step = criterion(logits, labels)

            tr_loss_step.backward()
            tr_loss_add += tr_loss_step

            if int(dist.get_rank()) in [2, 3, 6, 7]:
                assert tr_loss_step._is_initialized()
            else:
                assert not tr_loss_step._is_initialized()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                tr_loss_add /= self.gradient_accumulation_steps
                tr_loss = tr_loss_add

                optimizer.step()
                optimizer.clear_grad()
                lr_scheduler.step()

                tr_loss_add = 0
        return np.array(tr_loss)

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
            batch_size=2 * self.gradient_accumulation_steps,
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

            strategy.gradient_merge.enable = True
            strategy.gradient_merge.k_steps = self.gradient_accumulation_steps
            strategy.gradient_merge.avg = False

        dist_model = dist.to_static(
            model, dist_loader, criterion, optimizer, strategy=strategy
        )

        dist_model.train()
        loss = None
        for step, inputs in enumerate(dist_loader()):
            if step >= self.run_step_dy2static:
                break

            input_ids, labels = inputs
            loss = dist_model(input_ids, labels)

            lr_scheduler.step()
            if int(dist.get_rank()) in [2, 3, 6, 7]:
                assert loss is not None
            else:
                assert loss is None

        if loss is not None:
            loss = np.average(loss)
        return np.array(loss)

    def run_test_cases(self):
        self.init_dist_env()
        if self.gradient_accumulation_steps > 1:
            dy_losses = self.run_dynamic()
            self.init_dist_env()
            st_losses = self.run_dy2static()
            if int(dist.get_rank()) in [2, 3, 6, 7]:
                np.testing.assert_allclose(dy_losses, st_losses, atol=1e-7)

        else:
            dy_losses = self.run_llama(to_static=0)
            self.init_dist_env()
            st_losses = self.run_llama(to_static=1)
            assert len(dy_losses) == len(st_losses)
            for idx in range(len(dy_losses)):
                np.testing.assert_allclose(
                    dy_losses[idx], st_losses[idx], atol=1e-7
                )


if __name__ == '__main__':
    TestLlamaAuto().run_test_cases()
