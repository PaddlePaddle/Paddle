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
import logging
import os
import sys

import numpy as np

sys.path.append("../auto_parallel/hybrid_strategy")
from parallel_api import (
    RandomDataset,
    TestParallelAPI,
    get_mesh,
)
from single_llama_model import (
    LlamaForCausalLM,
    LlamaPretrainingCriterion,
)

import paddle
import paddle.distributed as dist
from paddle import LazyGuard
from paddle.io import BatchSampler, DataLoader


class TestParallelOnXPU(TestParallelAPI):
    def __init__(self):
        self.test_name = os.getenv("test_name")
        TestParallelAPI.__init__(self)

    def check_loss(self, loss):
        pretrained_loss = {}
        pretrained_loss['dp2mp1pp1'] = np.array(
            [9.080904, 9.06618], dtype=np.float32
        )
        pretrained_loss['dp1mp2pp1'] = np.array(
            [9.097355, 9.057393], dtype=np.float32
        )
        loss = np.array(loss, dtype=np.float32)
        if pretrained_loss.get(self.test_name) is not None:
            np.testing.assert_allclose(
                loss, pretrained_loss[self.test_name], atol=1e-04
            )

    def run_llama(self, to_static=0):
        if self.config.use_lazy_init:
            with LazyGuard():
                model = LlamaForCausalLM(
                    self.config, self.share_embedding, self.position_embedding
                )
        else:
            model = LlamaForCausalLM(
                self.config, self.share_embedding, self.position_embedding
            )
        model, optimizer, lr_scheduler = self.parallel_model(model)

        criterion = LlamaPretrainingCriterion(self.config)

        if self.config.use_lazy_init:
            for param in model.parameters():
                assert not param._is_initialized()
                param.initialize()

        if self.amp and not to_static:
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=self.amp_level,
                dtype=self.amp_dtype,
                master_grad=self.amp_master_grad,
            )

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

        if self.pp == 1:
            meshes = [get_mesh(0)]
        elif self.pp > 1:
            meshes = [get_mesh(0), get_mesh(-1)]
        else:
            raise ValueError("pp should be greater or equal to 1")

        dist_loader = dist.shard_dataloader(
            dataloader=train_dataloader,
            meshes=meshes,
            shard_dims="dp",
        )
        global_step = 1
        tr_loss = float(0)

        if not to_static:
            model.train()
            scaler = None
            if self.amp and self.amp_dtype == "float16":
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                scaler = dist.shard_scaler(scaler)
            loss_data = []
            for step, inputs in enumerate(dist_loader()):
                input_ids, labels = inputs
                custom_black_list = [
                    "reduce_sum",
                    "c_softmax_with_cross_entropy",
                ]
                custom_white_list = []
                if self.amp_level == "O2":
                    custom_white_list.extend(
                        ["lookup_table", "lookup_table_v2"]
                    )
                with paddle.amp.auto_cast(
                    self.amp,
                    custom_black_list=set(custom_black_list),
                    custom_white_list=set(custom_white_list),
                    level=self.amp_level,
                    dtype=self.amp_dtype,
                ):
                    logits = model(input_ids)
                    tr_loss_step = criterion(logits, labels)
                if self.gradient_accumulation_steps > 1:
                    tr_loss_step /= self.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(tr_loss_step).backward()
                else:
                    tr_loss_step.backward()
                tr_loss += tr_loss_step
                if global_step % self.gradient_accumulation_steps == 0:
                    logging.info(
                        f"step: {global_step // self.gradient_accumulation_steps}  loss: {tr_loss._local_value().numpy()}"
                    )
                    loss_data.append(tr_loss._local_value().numpy()[0])
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.clear_grad()
                    lr_scheduler.step()
                    tr_loss = 0

                global_step += 1
                if global_step // self.gradient_accumulation_steps >= 3:
                    break
            self.check_loss(loss_data)
        else:
            strategy = dist.Strategy()
            if self.gradient_accumulation_steps > 1:
                strategy.pipeline.accumulate_steps = (
                    self.gradient_accumulation_steps
                )

            if self.amp:
                amp = strategy.amp
                amp.enable = self.amp
                amp.dtype = self.amp_dtype
                amp.level = self.amp_level.lower()
                if self.amp_master_grad:
                    amp.use_master_grad = True

            dist_model = dist.to_static(
                model,
                dist_loader,
                criterion,
                optimizer,
                strategy=strategy,
            )

            dist_model.train()
            for step, inputs in enumerate(dist_loader()):
                input_ids, labels = inputs
                loss = dist_model(input_ids, labels)
                logging.info(f"step: {step}  loss: {loss}")
                if step >= 3:
                    break

    def run_test_cases(self):
        # dynamic
        self.run_llama(0)
        # static
        # self.run_llama(1)


if __name__ == '__main__':
    TestParallelOnXPU().run_test_cases()
