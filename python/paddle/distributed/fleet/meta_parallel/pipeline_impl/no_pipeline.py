#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import paddle
from paddle import framework

from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters,
    broadcast_moe_sharding_parameters,
    broadcast_mp_parameters,
    broadcast_sep_parameters,
    broadcast_sharding_parameters,
)
from ..utils.log_util import logger
from .meta_parallel_base import MetaParallelBase
from .parallel_layers.pp_layers import PipelineLayer
from .utils import FakeMicroDataset, PipelineDatasetPreprocessor


class NoPipelineParallel(MetaParallelBase):
    def __init__(self, layers, strategy, hcg=None):
        assert isinstance(layers, PipelineLayer)
        super().__init__(layers, hcg, strategy)
        self._layers = layers
        self._strategy = strategy
        self._hcg = hcg

        self.micro_batch_size = self._strategy.pipeline_configs[
            "micro_batch_size"
        ]
        self.accumulate_steps = self._strategy.pipeline_configs[
            "accumulate_steps"
        ]
        self._delay_scale_loss = self._strategy.hybrid_configs[
            "pp_configs"
        ].delay_scale_loss
        self._dp_comm_overlap = False
        self._sharding_comm_overlap = False

        # store total loss of entire batch. It contains the loss of each micro batch in a list, then contains many loss_fn's list in total_loss.
        self.total_loss = None

        # default loss function index
        self.loss_fn_idx = 0

        if self._hcg is not None:
            self.use_data_parallel = (
                self._hcg.get_data_parallel_world_size() > 1
            )
            self.use_model_parallel = (
                self._hcg.get_model_parallel_world_size() > 1
            )
            self.use_sep_parallel = self._hcg.get_sep_parallel_world_size() > 1
            self.use_sharding_parallel = (
                self._hcg.get_sharding_parallel_world_size() > 1
            )
            self.use_moe_sharding_parallel = (
                self._hcg.get_moe_sharding_parallel_world_size() > 1
            )

            self.dp_group = self._hcg.get_data_parallel_group()
            # fused sep and dp
            if self.use_sep_parallel:
                self.dp_group = self._hcg.get_dp_sep_parallel_group()

            if self.use_model_parallel:
                logger.info("start broadcast mp parameters")
                broadcast_mp_parameters(self._layers, self._hcg)

            if self.use_sep_parallel:
                logger.info("start broadcast sep parameters")
                broadcast_sep_parameters(self._layers, self._hcg)

            if self.use_sharding_parallel:
                logger.info("start broadcast sharding parameters")
                broadcast_sharding_parameters(self._layers, self._hcg)

            if self.use_data_parallel:
                logger.info("start broadcast dp parameters")
                broadcast_dp_parameters(self._layers, self._hcg)

            if self.use_moe_sharding_parallel:
                logger.info("start broadcast moe_sharding parameters")
                broadcast_moe_sharding_parameters(self._layers, self._hcg)

    def is_pipeline_last_stage(self, ignore_virtual=False):
        return True

    def _check_micro_batch_data_valid(self, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            for data in micro_batch_data:
                self._check_micro_batch_data_valid(data)
        elif isinstance(micro_batch_data, dict):
            for value in micro_batch_data.values():
                self._check_micro_batch_data_valid(value)
        elif micro_batch_data is not None:
            assert isinstance(micro_batch_data, paddle.Tensor)

    def _prepare_training(self, data, optimizer, lr_scheduler):
        assert framework._dygraph_tracer()._has_grad, (
            "Please enable the generation of gradients."
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._layers.train()
        return data

    def _optimizer_step(self):
        if self._delay_scale_loss:
            for p in self._layers.parameters():
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    assert p.grad is None
                    p.main_grad = p.main_grad.scale(1.0 / self.accumulate_steps)
                elif p.grad is not None:
                    p.grad = p.grad.scale(1.0 / self.accumulate_steps)

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.clear_grad()

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def forward_backward_pipeline(
        self,
        data,
        scaler=None,
        return_micro_batch_loss=False,
    ):
        self.scaler = scaler
        self.total_loss = None

        if isinstance(data, PipelineDatasetPreprocessor):
            data = data()

        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            micro_dataset = data
        else:
            micro_dataset = FakeMicroDataset(
                data,
                True,
                True,
                self.accumulate_steps,
                self.micro_batch_size,
            )

        loss_list = []
        for _ in range(self.accumulate_steps):
            # data prepare
            data_iter = next(micro_dataset)
            input_tensor = data_iter[0]
            label = data_iter[1]
            self._check_micro_batch_data_valid(input_tensor)
            self._check_micro_batch_data_valid(label)

            # forward
            output_tensor = self._layers.forward(input_tensor)

            # loss is loss_fn[loss_fn_idx]'s result
            loss = None
            # cal loss
            for idx, loss_fn in enumerate(self._layers._loss_fn):
                loss_tensor = loss_fn(output_tensor, label)
                assert isinstance(loss_tensor, paddle.Tensor), (
                    "Currently, loss_fn should obtain Paddle.Tensor dtype"
                )
                with paddle.amp.auto_cast(enable=False):
                    if self.accumulate_steps > 1 and not self._delay_scale_loss:
                        loss_tensor = loss_tensor / self.accumulate_steps
                if self.total_loss is None:
                    self.total_loss = []
                # when self.total_loss length is less than idx, append a new tensor
                if len(self.total_loss) <= idx:
                    self.total_loss.append([])

                self.total_loss[idx].append(loss_tensor.detach())

                if idx == self.loss_fn_idx:
                    loss = loss_tensor

            # backward
            with paddle.amp.auto_cast(enable=False):
                if self.scaler:
                    paddle.autograd.backward(self.scaler.scale(loss))
                else:
                    paddle.autograd.backward(loss)

            assert self.total_loss is not None, (
                "train_batch() in last stage should obtain valid loss"
            )

        losses = []
        with paddle.amp.auto_cast(enable=False):
            for idx in range(len(self._layers._loss_fn)):
                self.total_loss[idx] = paddle.to_tensor(self.total_loss[idx])
                if not return_micro_batch_loss:
                    # TODO(shenliang03): it will use mean/sum to calculate loss
                    tmp = paddle.zeros_like(self.total_loss[idx][0])
                    for loss in self.total_loss[idx]:
                        tmp += loss.detach()
                    if not self._delay_scale_loss:
                        losses.append(tmp)
                    else:
                        losses.append(tmp / self.accumulate_steps)
                else:
                    losses.append(self.total_loss[idx].detach())
        return losses[0] if len(losses) == 1 else losses

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
        loss_fn_idx=0,
        return_micro_batch_loss=False,
    ):
        data = self._prepare_training(data, optimizer, lr_scheduler)

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        # no pipeline parallel
        train_loss = self.forward_backward_pipeline(
            data, scaler, return_micro_batch_loss=return_micro_batch_loss
        )

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False, loss_fn_idx=0):
        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        self.total_loss = None

        if isinstance(data, PipelineDatasetPreprocessor):
            data = data()

        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            micro_dataset = data
        else:
            micro_dataset = FakeMicroDataset(
                data,
                True,
                True,
                self.accumulate_steps,
                self.micro_batch_size,
            )

        loss_list = []
        for _ in range(self.accumulate_steps):
            # data prepare
            data_iter = next(micro_dataset)
            input_tensor = data_iter[0]
            label = data_iter[1]
            self._check_micro_batch_data_valid(input_tensor)
            self._check_micro_batch_data_valid(label)

            # forward
            output_tensor = self._layers.forward(input_tensor)

            # loss is loss_fn[loss_fn_idx]'s result
            loss = None
            # cal loss
            for idx, loss_fn in enumerate(self._layers._loss_fn):
                loss_tensor = loss_fn(output_tensor, label)
                assert isinstance(loss_tensor, paddle.Tensor), (
                    "Currently, loss_fn should obtain Paddle.Tensor dtype"
                )
                with paddle.amp.auto_cast(enable=False):
                    if self.accumulate_steps > 1 and not self._delay_scale_loss:
                        loss_tensor = loss_tensor / self.accumulate_steps
                if self.total_loss is None:
                    self.total_loss = []
                # when self.total_loss length is less than idx, append a new tensor
                if len(self.total_loss) <= idx:
                    self.total_loss.append([])

                self.total_loss[idx].append(loss_tensor.detach())

                if idx == self.loss_fn_idx:
                    loss = loss_tensor

            assert self.total_loss is not None, (
                "train_batch() in last stage should obtain valid loss"
            )

        losses = []
        return_micro_batch_loss = False
        for idx in range(len(self._layers._loss_fn)):
            self.total_loss[idx] = paddle.to_tensor(self.total_loss[idx])
            if not return_micro_batch_loss:
                # TODO(shenliang03): it will use mean/sum to calculate loss
                tmp = paddle.zeros_like(self.total_loss[idx][0])
                for loss in self.total_loss[idx]:
                    tmp += loss.detach()
                if not self._delay_scale_loss:
                    losses.append(tmp)
                else:
                    losses.append(tmp / self.accumulate_steps)
            else:
                losses.append(self.total_loss[idx].detach())
        return losses[0] if len(losses) == 1 else losses
