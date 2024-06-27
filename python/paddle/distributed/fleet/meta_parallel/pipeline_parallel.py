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

import os
import queue
import sys
from collections import defaultdict

import paddle
from paddle import framework
from paddle.fluid.core import set_nccl_channels

from ..meta_optimizers.dygraph_optimizer import HybridParallelOptimizer
from ..utils import timer_helper as timer
from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters,
    broadcast_mp_parameters,
    broadcast_sharding_parameters,
)
from ..utils.log_util import logger
from ..utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
)
from .meta_parallel_base import MetaParallelBase
from .parallel_layers.pp_layers import PipelineLayer
from .pp_utils import p2p_communication as p2p

__all__ = []

g_shard_use_reduce = int(os.environ.get("FLAGS_shard_use_reduce", 1))
g_enable_nccl_channel_shrink = int(
    os.environ.get("FLAGS_enable_nccl_channel_shrink", 0)
)
nccl_channel_num_for_overlap_comm = 1 if g_enable_nccl_channel_shrink else 0


def get_action(is_dp, shard_split_param=False):
    if is_dp or not g_shard_use_reduce:
        return HOOK_ACTION.ALL_REDUCE
    if shard_split_param:
        return HOOK_ACTION.REDUCE_SCATTER
    return HOOK_ACTION.REDUCE


# assume only the first stage and last stage need data, and data consumption are ordred;
# to be replaced by real micro dataset from reader
class FakeMicroDataset:
    def __init__(
        self, data, is_first_stage, is_last_stage, acc_steps, micro_batch_size
    ):
        self._data = data
        self._index = 0
        self._acc_steps = acc_steps
        self._is_first_stage = is_first_stage
        self._is_last_stage = is_last_stage
        self._micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        assert self._index < self._acc_steps
        assert self._is_first_stage or self._is_last_stage
        micro_batch_data = self._load_micro_batch(self._index)
        self._index += 1
        return micro_batch_data

    def _load_micro_batch(self, micro_step):
        inputs = self._data

        data = None
        label = None
        if self._is_first_stage:
            assert len(inputs) == 2, "length of input should be 2"
            data = self._load_micro_batch_impl(inputs[0], micro_step)

        if self._is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            label = self._load_micro_batch_impl(inputs[1], micro_step)

        return (data, label)

    def _load_micro_batch_impl(self, inputs, micro_step):
        begin = micro_step * self._micro_batch_size
        end = begin + self._micro_batch_size

        if isinstance(inputs, tuple):
            output = []
            for data in inputs:
                if isinstance(data, list):
                    assert (
                        len(data) == self._acc_steps
                    ), "length of data should be %d, but it is %d" % (
                        self._acc_steps,
                        len(data),
                    )
                    output.append(
                        data[micro_step].detach()
                        if data[micro_step] is not None
                        else None
                    )
                elif data is not None:
                    self._check_data_vaild(data)
                    output.append(data[begin:end, :].detach())
                else:
                    output.append(None)
            return tuple(output)

        elif isinstance(inputs, list):
            assert (
                len(inputs) == self._acc_steps
            ), "length of data should be %d, but it is %d" % (
                self.accumulate_steps,
                len(inputs),
            )
            return inputs[micro_step].detach()
        elif inputs is not None:
            self._check_data_vaild(inputs)
            return inputs[begin:end, :].detach()
        else:
            return None

    def _check_data_vaild(self, data):
        batch_size = data.shape[0]
        assert self._micro_batch_size * self._acc_steps == batch_size, (
            "batch_size needs to be divisible by micro_batch_size. Currently, "
            "batch_size = %d, micro_batch_size = %d, accumulate_steps = %d."
            % (batch_size, self._micro_batch_size, self._acc_steps)
        )


class PipelineParallel(MetaParallelBase):
    def __init__(self, layers, hcg, strategy):
        if not isinstance(layers, PipelineLayer):
            raise TypeError(
                "The Layer should be a derived class of PipelineLayer."
            )
        super().__init__(layers, hcg, strategy)
        self.use_data_parallel = self._hcg.get_data_parallel_world_size() > 1
        self.use_model_parallel = self._hcg.get_model_parallel_world_size() > 1
        self.use_sharding_parallel = (
            self._hcg.get_sharding_parallel_world_size() > 1
        )

        self.total_loss = None

        self.micro_batch_size = self._strategy.pipeline_configs[
            'micro_batch_size'
        ]
        self.accumulate_steps = self._strategy.pipeline_configs[
            'accumulate_steps'
        ]
        # If sent tensor are not the same from different hosts,
        # they shouldn't been sent partially and then concated as a whole tensor.
        self._enable_partial_send_recv = self._strategy.pipeline_configs[
            'enable_partial_send_recv'
        ]
        self._using_cache = self._strategy.pipeline_configs['p2p_cache_shape']

        self.num_stages = self._hcg.get_pipe_parallel_world_size()
        self.stage_id = self._hcg.get_stage_id()
        self.pp_group = self._hcg.get_pipe_parallel_group()
        self.dp_group = self._hcg.get_data_parallel_group()
        self.sharding_group = self._hcg.get_sharding_parallel_group()

        self._virtual_pp_world_size = None
        self._virtual_pp_rank = None
        self._real_pp_world_size = self.num_stages
        self._real_pp_rank = self.stage_id

        self._delay_scale_loss = self._strategy.hybrid_configs[
            "pp_configs"
        ].delay_scale_loss
        # TODO(PP Dev): support dp_comm_overlap without use_main_grad training.
        # This combination will trigger inplace check error during `reshape_` in funct `_split_tensors`.
        self._dp_comm_overlap = self._strategy.hybrid_configs[
            "pp_configs"
        ].dp_comm_overlap
        self._sharding_comm_overlap = self._strategy.hybrid_configs[
            "pp_configs"
        ].sharding_comm_overlap
        self._enable_timer = self._strategy.hybrid_configs[
            "pp_configs"
        ].enable_timer
        self._release_gradients = self._strategy.hybrid_configs[
            "pp_configs"
        ].release_gradients

        self._sharding_split_param = self._strategy.hybrid_configs[
            "sharding_configs"
        ].split_param

        self._overlap_p2p_comm = self._strategy.hybrid_configs[
            "pp_configs"
        ].overlap_p2p_comm

        self._clear_every_step_cache = self._strategy.hybrid_configs[
            "pp_configs"
        ].clear_every_step_cache

        self._batch_p2p_comm = not self._overlap_p2p_comm

        if self._dp_comm_overlap:
            assert self.use_data_parallel and self.num_stages > 1

        if self._sharding_comm_overlap:
            assert self.use_sharding_parallel and self.num_stages > 1

        assert not (
            self._dp_comm_overlap and self._sharding_comm_overlap
        ), "Cannot use dp pp overlap and sharding pp overlap at the same time."

        self._chunk_2_comm_buffers = defaultdict(list)
        self._comm_overlap = (
            self._dp_comm_overlap or self._sharding_comm_overlap
        )

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self.timers = timer.get_timers()

        p2p.initialize_p2p_groups(
            hcg,
            self._enable_partial_send_recv,
            self._enable_timer,
        )

        # construct pipeline meta info
        self._p2p_helper = p2p.P2pHelper(self._using_cache)

        self.global_rank = self._hcg.get_global_rank()
        self.micro_batch_id = 0

        self._compute_loss = True

        logger.info(
            "Pipeline Info -- num_stages: {}, stage_id: {}".format(
                self.num_stages, self.stage_id
            )
        )

        if self.use_model_parallel:
            logger.info("start broadcast mp parameters")
            broadcast_mp_parameters(self._layers, self._hcg)

        if self.use_sharding_parallel:
            logger.info("start broadcast sharding parameters")
            broadcast_sharding_parameters(self._layers, self._hcg)

        if self.use_data_parallel:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

        if self._dp_comm_overlap:
            self.register_allreduce_overlap_hook(
                self._layers, self.dp_group, self.accumulate_steps, True
            )

    def is_pipeline_first_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != 0:
                    return False
        assert self._real_pp_rank is not None
        return self._real_pp_rank == 0

    def is_pipeline_last_stage(self, ignore_virtual=False):
        if not ignore_virtual:
            if self._virtual_pp_world_size is not None:
                assert self._virtual_pp_rank is not None
                if self._virtual_pp_rank != (self._virtual_pp_world_size - 1):
                    return False
        assert self._real_pp_rank is not None
        assert self._real_pp_world_size is not None
        return self._real_pp_rank == (self._real_pp_world_size - 1)

    def set_virtual_pipeline_rank(self, rank):
        self._virtual_pp_rank = rank

    def fused_gradient(
        self, model, comm_group, acc_steps, dp, group_size=128 * 1024 * 1024
    ):
        if model.get_num_virtual_stages() > 1:
            models = model.get_model_chunks()
        else:
            models = [model]

        act = get_action(dp, self._sharding_split_param)

        if act == HOOK_ACTION.REDUCE:
            assert hasattr(self, "optimizer")
            assert hasattr(self.optimizer, "_param2rank")
            _param2rank = self.optimizer._param2rank

        for chunk_idx, model in enumerate(models):
            # For virtual pipeline. Will separate parameters in different chunk into
            # different groups to get the best performance.

            fused_parameter_group = {}

            parameter_list = [
                p for p in model.parameters() if not p.stop_gradient
            ]
            if len(parameter_list) < 1:
                return

            if act == HOOK_ACTION.REDUCE:
                # Sort parameters for sharding, since they have different dst rank
                for p in parameter_list:
                    assert p.name in _param2rank
                    dst_rank = _param2rank[p.name]
                    if dst_rank in fused_parameter_group:
                        fused_parameter_group[dst_rank].append(p)
                    else:
                        fused_parameter_group[dst_rank] = [p]
            else:
                fused_parameter_group[-1] = parameter_list

            for dst in fused_parameter_group:
                parameter_list = fused_parameter_group[dst]
                if act == HOOK_ACTION.REDUCE:
                    # parse the relative dst rank to absolute dst rank for sharding
                    dst = comm_group.ranks[dst]

                var_groups = assign_group_by_size(parameter_list, group_size)
                for group_idx, parameters in var_groups.items():
                    buffer = FusedCommBuffer(
                        group_idx,
                        parameters,
                        comm_group,
                        acc_steps,
                        act,
                        dst,
                        release_grads=self._release_gradients,
                    )
                    self._chunk_2_comm_buffers[chunk_idx].append(buffer)

        return self._chunk_2_comm_buffers

    def bw_hook_func(self, buffer, param):
        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            buffer.add_grad(param)

        return fused_allreduce

    def register_allreduce_overlap_hook(
        self, model, comm_group, acc_steps, dp, group_size=128 * 1024 * 1024
    ):
        # register hook
        self.fused_gradient(model, comm_group, acc_steps, dp, group_size)
        for _, buffers in self._chunk_2_comm_buffers.items():
            for buffer in buffers:
                for param in buffer._params:
                    param._register_backward_hook(
                        self.bw_hook_func(buffer, param)
                    )

    def timer_printer(self):
        if not self._enable_timer:
            return
        all_flag_names = self.timers.timers.keys()
        self.timers.log(all_flag_names)

    def forward_backward_pipeline(self, data, scaler=None):
        # use the 1f1b scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py

        self.scaler = scaler

        # store total loss of entire batch
        self.total_loss = None

        # store data id for micro_batch
        self.micro_batch_id = 0

        startup_steps = self.num_stages - self.stage_id - 1
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        micro_dataset = self._wrap_data(data)

        for step_id in range(startup_steps):
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage()
            )

            output_tensor = self._forward_step(input_tensor, micro_dataset)
            self._p2p_helper.send_forward(
                output_tensor, self.is_pipeline_last_stage()
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)

        if steady_steps > 0:
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage()
            )

        for i in range(steady_steps):
            last_iter = i == (steady_steps - 1)

            output_tensor = self._forward_step(input_tensor, micro_dataset)

            output_tensor_grad = self._p2p_helper.send_forward_recv_backward(
                output_tensor, self.is_pipeline_last_stage()
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)

            input_tensor, output_tensor = input_buffers.pop(
                0
            ), output_buffers.pop(0)

            input_tensor_grad = self._backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )

            if last_iter:
                input_tensor = None
                self._p2p_helper.send_backward(
                    input_tensor_grad, self.is_pipeline_first_stage()
                )
            else:
                input_tensor = self._p2p_helper.send_backward_recv_forward(
                    input_tensor_grad, self.is_pipeline_first_stage()
                )

        for i in range(startup_steps):
            input_tensor = input_buffers.pop(0)
            output_tensor = output_buffers.pop(0)

            output_tensor_grad = self._p2p_helper.recv_backward(
                self.is_pipeline_last_stage()
            )

            input_tensor_grad = self._backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )
            self._p2p_helper.send_backward(
                input_tensor_grad, self.is_pipeline_first_stage()
            )

        if self._comm_overlap:
            assert (
                len(self._chunk_2_comm_buffers) > 0
            ), "comm buffers should be created"
            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer.scale_and_split_grads()

        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").start()
        self._layers.allreduce_shared_weight_gradients()
        if self._enable_timer:
            self.timers("allreduce_shared_weight_gradients").stop()
            self.timers("broadcast_final_loss").start()
        with paddle.amp.auto_cast(enable=False):
            train_loss = self._broadcast_final_loss()
        if self._enable_timer:
            self.timers("broadcast_final_loss").stop()

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()
        return train_loss

    def register_sharding_comm_overlap_hook(self, optimizer):
        """for delayed hook register until we get optimizer"""
        assert isinstance(
            optimizer, HybridParallelOptimizer
        ), 'optimizer should be HybridParallelOptimizer subclass.'
        self.optimizer = optimizer
        if self._sharding_comm_overlap and len(self._chunk_2_comm_buffers) == 0:
            self.register_allreduce_overlap_hook(
                self._layers, self.sharding_group, self.accumulate_steps, False
            )

    def _prepare_training(self, data, optimizer, lr_scheduler):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        assert isinstance(
            optimizer, HybridParallelOptimizer
        ), 'optimizer should be HybridParallelOptimizer subclass.'

        assert (
            framework._dygraph_tracer()._has_grad
        ), 'Please enable the generation of gradients.'

        if self.is_pipeline_first_stage(
            ignore_virtual=True
        ) or self.is_pipeline_last_stage(ignore_virtual=True):
            assert (
                data is not None
            ), "For the first and the last stage, the data must be set."
        else:
            data = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._layers.train()

        # delayed hook register until we get optimizer
        self.register_sharding_comm_overlap_hook(optimizer)

        return data

    def _wrap_data(self, data):
        """
        for backward compatibilty, wrap data to Fake FakeMicroDataset if it is of type list or tuple
        """
        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            return data

        micro_dataset = FakeMicroDataset(
            data,
            self.is_pipeline_first_stage(ignore_virtual=True),
            self.is_pipeline_last_stage(ignore_virtual=True),
            self.accumulate_steps,
            self.micro_batch_size,
        )
        return micro_dataset

    def train_batch(self, data, optimizer, lr_scheduler=None, scaler=None):
        data = self._prepare_training(data, optimizer, lr_scheduler)
        # 1f1b scheduler for pipeline parallel
        train_loss = self.forward_backward_pipeline(data, scaler)

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        self._compute_loss = compute_loss

        # store data id for micro_batch
        self.micro_batch_id = 0

        # store total loss of entire batch
        self.total_loss = None

        startup_steps = self.num_stages - self.stage_id - 1
        startup_steps = min(startup_steps, self.accumulate_steps)
        steady_steps = self.accumulate_steps - startup_steps

        input_buffers = []
        output_buffers = []

        micro_dataset = self._wrap_data(data)

        for step_id in range(startup_steps):
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage()
            )

            output_tensor = self._forward_step(input_tensor, micro_dataset)
            self._p2p_helper.send_forward(
                output_tensor, self.is_pipeline_last_stage()
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage()
            )

        for i in range(steady_steps):
            last_iter = i == (steady_steps - 1)

            output_tensor = self._forward_step(input_tensor, micro_dataset)
            self._p2p_helper.send_forward(
                output_tensor, self.is_pipeline_last_stage()
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not last_iter:
                input_tensor = self._p2p_helper.recv_forward(
                    self.is_pipeline_first_stage()
                )

        if self._compute_loss:
            self.train_loss = self._broadcast_final_loss()
        else:
            self.train_loss = output_buffers

        return self.train_loss

    def _forward_step(self, input_tensor, micro_dataset, chunk_id=None):
        if self._enable_timer:
            self.timers("forward_step").start()
        if self.is_pipeline_first_stage():
            input_tensor = next(micro_dataset)[0]
            self._check_micro_batch_data_valid(input_tensor)

        assert chunk_id is None or isinstance(chunk_id, int)

        output_tensor = self._layers.forward(input_tensor, chunk_id=chunk_id)

        if self.is_pipeline_last_stage():
            # train calculate loss for train
            if self._compute_loss:
                assert (
                    self._layers._loss_fn is not None
                ), "loss function should exist to compute loss"
                labels = next(micro_dataset)[1]
                self._check_micro_batch_data_valid(labels)
                output_tensor = self._layers._loss_fn(output_tensor, labels)
                assert isinstance(
                    output_tensor, (paddle.Tensor, framework.core.eager.Tensor)
                ), "Currently, loss_fn should obtain Paddle.Tensor dtype"

                with paddle.amp.auto_cast(enable=False):
                    if self.accumulate_steps > 1 and not self._delay_scale_loss:
                        output_tensor = output_tensor / self.accumulate_steps

                    if self.total_loss is None:
                        self.total_loss = paddle.zeros_like(output_tensor)
                    self.total_loss += output_tensor.detach()

        if self.is_pipeline_first_stage() or self.is_pipeline_last_stage():
            # Only increase micro batch id at virtual first/last pp stage.
            # The micro batch id is used to load data, therefore, only increase it when load data.
            self.micro_batch_id += 1
        if self._enable_timer:
            self.timers("forward_step").stop()
        return output_tensor

    def _check_micro_batch_data_valid(self, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            for data in micro_batch_data:
                self._check_micro_batch_data_valid(data)
        elif micro_batch_data is not None:
            micro_batch_size = micro_batch_data.shape[0]
            assert (
                micro_batch_size == self.micro_batch_size
            ), f"expected micro_batch_size {self.micro_batch_size} but get {micro_batch_size}"

    def _backward_step(self, input_tensor, output_tensor, output_tensor_grad):
        if self._enable_timer:
            self.timers("backward_step").start()
        with paddle.amp.auto_cast(enable=False):
            if self.is_pipeline_last_stage():
                assert output_tensor_grad is None
                if self.scaler:
                    paddle.autograd.backward(self.scaler.scale(output_tensor))
                else:
                    paddle.autograd.backward(output_tensor)
            else:
                if isinstance(output_tensor, tuple):
                    outputs = [t for t in output_tensor if not t.stop_gradient]
                    assert len(outputs) == len(output_tensor_grad)
                    paddle.autograd.backward(
                        tensors=outputs,
                        grad_tensors=list(output_tensor_grad),
                    )
                else:
                    paddle.autograd.backward(
                        tensors=[output_tensor],
                        grad_tensors=[output_tensor_grad],
                    )

            input_tensor_grad = None
            if input_tensor is not None:
                if isinstance(input_tensor, tuple):
                    input_tensor_grad = tuple(
                        [t.grad for t in input_tensor if not t.stop_gradient]
                    )
                else:
                    input_tensor_grad = input_tensor.grad
            if self._enable_timer:
                self.timers("backward_step").stop()
            return input_tensor_grad

    def _broadcast_final_loss(self):
        # Since the last backward run in interleave will set the virtual rank to 0,
        # here we need to check last stage ignoring virtual stage.
        if self.is_pipeline_last_stage(ignore_virtual=True):
            assert (
                self.total_loss is not None
            ), "train_batch() in last stage should obtain vaild loss"
            loss = (
                self.total_loss.detach()
                if not self._delay_scale_loss
                else self.total_loss / self.accumulate_steps
            )
            is_fp32 = (
                paddle.full([], 1, 'int64')
                if loss.dtype == paddle.float32
                else paddle.full([], 0, 'int64')
            )
            paddle.distributed.broadcast(
                is_fp32, src=self.global_rank, sync_op=True, group=self.pp_group
            )
            paddle.distributed.broadcast(
                loss, src=self.global_rank, sync_op=True, group=self.pp_group
            )
        else:
            is_fp32 = paddle.full([], 1, 'int64')
            paddle.distributed.broadcast(
                is_fp32,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                sync_op=True,
                group=self.pp_group,
            )
            loss = (
                paddle.zeros(shape=[1], dtype="float32")
                if is_fp32.item()
                else paddle.zeros(shape=[1], dtype="float16")
            )
            paddle.distributed.broadcast(
                loss,
                src=self._hcg.get_rank_from_stage(self.num_stages - 1),
                sync_op=True,
                group=self.pp_group,
            )
        return loss

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

        if self._release_gradients:
            self.optimizer.clear_grad(set_to_zero=False)
            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer._clear_grad_storage()
        else:
            self.optimizer.clear_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _release_output(self, output):
        def can_free(t):
            return (
                t is not None
                and isinstance(t, paddle.Tensor)
                and t._is_initialized()
                and t.inplace_version == 0
            )

        if isinstance(output, (tuple, list)):
            for t in output:
                if can_free(t):
                    t._clear_dataptr()

        elif can_free(output):
            output._clear_dataptr()


class PipelineParallelWithInterleave(PipelineParallel):
    # pipeline parallel with interleave scheduler

    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        assert layers.get_num_virtual_stages() > 1
        self._check_sanity()
        self.num_model_chunks = layers.get_num_virtual_stages()
        self.model_chunks = layers.get_model_chunks()
        assert self.model_chunks is not None
        assert len(self.model_chunks) == self.num_model_chunks
        self._virtual_pp_world_size = self.num_model_chunks
        self._virtual_pp_rank = 0
        self._assign_vpp_info(self.model_chunks)

    def _check_sanity(self):
        assert (
            framework.in_dygraph_mode()
        ), "virtual pipeline stage with interleave only support eager dygraph mode"

        assert (
            self.num_stages > 2
        ), "virtual pipeline must run under pp degree > 2"

        assert (
            self.accumulate_steps >= 2 * self.num_stages
        ), "accumulate_steps({}) should be greater than or equal to 2 * num_stages({}) for pipeline with interleave".format(
            self.accumulate_steps, self.num_stages
        )

    def _assign_vpp_info(self, chunks):
        chunk_num = len(chunks)
        for i, chunk in enumerate(chunks):
            for p in chunk.parameters():
                p._chunk_info = {"chunk_id": i, "chunk_num": chunk_num}

    def _get_virtual_pp_rank(self, micro_step, forward):

        first_chunk_acc = (
            self.accumulate_steps % self.num_stages + self.num_stages
        )
        first_chunk_steps = first_chunk_acc * self.num_model_chunks

        if micro_step < first_chunk_steps:
            virtual_pp_stage = micro_step // first_chunk_acc
        else:
            micro_step -= first_chunk_steps
            virtual_pp_stage = micro_step % (
                self.num_stages * self.num_model_chunks
            )
            virtual_pp_stage = virtual_pp_stage // self.num_stages

        if not forward:
            virtual_pp_stage = self.num_model_chunks - virtual_pp_stage - 1
        return virtual_pp_stage

    def _forward_step_helper(self, micro_dataset, micro_step):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=True)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        if not self._forward_only:
            assert hasattr(self, 'output_tensor_grads')

        if self.is_pipeline_first_stage():
            if len(self.input_tensors[virtual_pp_rank]) == len(
                self.output_tensors[virtual_pp_rank]
            ):
                self.input_tensors[virtual_pp_rank].append(None)
        input_tensor = self.input_tensors[virtual_pp_rank][-1]
        output_tensor = self._forward_step(
            input_tensor, micro_dataset, virtual_pp_rank
        )
        self.output_tensors[virtual_pp_rank].append(output_tensor)

        if self._forward_only:
            # no need to store tensor for backward
            self.input_tensors[virtual_pp_rank].pop()
            self.output_tensors[virtual_pp_rank].pop()

        return output_tensor

    def _overlap_comm_grads(self):
        if self._comm_overlap:
            self._backward_step_count += 1
            sync_step = self._backward_step_count - self.stage_id
            if sync_step > 0 and sync_step % self.num_stages == 0:
                chunk_idx = self._virtual_pp_world_size - (
                    sync_step // self.num_stages
                )
                old_value = set_nccl_channels(nccl_channel_num_for_overlap_comm)
                for buffer in self._chunk_2_comm_buffers[chunk_idx]:
                    buffer.comm_grads()
                set_nccl_channels(old_value)

            if self.stage_id != 0:
                if (
                    self._backward_step_count
                    == self.num_stages * self.num_model_chunks
                ):
                    old_value = set_nccl_channels(
                        nccl_channel_num_for_overlap_comm
                    )
                    for buffer in self._chunk_2_comm_buffers[0]:
                        buffer.comm_grads()
                    set_nccl_channels(old_value)

    def _sync_overlap_grads(self):
        if self._comm_overlap:
            assert (
                self._backward_step_count
                == self.num_stages * self.num_model_chunks
            ), (
                "backward step count should be equal to accumulate steps * virtual pp world size,"
                f" but get {self._backward_step_count}, excepted result is {self.num_stages * self.num_model_chunks}"
            )

            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer.scale_and_split_grads()

    def _backward_step_helper(self, micro_step):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=False)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        assert hasattr(self, 'output_tensor_grads')

        if self.is_pipeline_last_stage():
            if len(self.output_tensor_grads[virtual_pp_rank]) == 0:
                self.output_tensor_grads[virtual_pp_rank].append(None)

        input_tensor = self.input_tensors[virtual_pp_rank].pop(0)
        output_tensor = self.output_tensors[virtual_pp_rank].pop(0)
        output_tensor_grad = self.output_tensor_grads[virtual_pp_rank].pop(0)
        input_tensor_grad = self._backward_step(
            input_tensor, output_tensor, output_tensor_grad
        )

        self._overlap_comm_grads()

        return input_tensor_grad

    def bw_hook_func(self, buffer, param):
        # For pipeline with interleave, we need to add grad to buffer without communication.
        # Use communication where appropriate to avoid dp communication and pp scheduling conflicts.
        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            buffer.add_grad(param, use_comm=False)

        return fused_allreduce

    def register_allreduce_overlap_hook(self, model, comm_group, acc_steps, dp):
        super().register_allreduce_overlap_hook(
            model, comm_group, acc_steps, dp, group_size=sys.maxsize
        )

    def forward_backward_pipeline(
        self, data, scaler, forward_only=False, compute_loss=True
    ):
        # use interleave scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py
        if not compute_loss:
            assert (
                not forward_only
            ), "compute_loss can only be set to False when forward_only is set to True"

        # NOTE(shenliang03): Due to ring_exchange for pipeline with interleave, cache should be enabled
        assert (
            self._using_cache
        ), "cache should be enabled for pipeline with interleave"

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only

        first_chunk_acc = (
            self.accumulate_steps % self.num_stages + self.num_stages
        )
        first_chunk_steps = first_chunk_acc * self.num_model_chunks
        fwd_buffer_queue = queue.Queue()
        bwd_buffer_queue = queue.Queue()
        skip_steps = self.accumulate_steps % self.num_stages

        left_id = skip_steps
        right_id = left_id + first_chunk_acc * (self.num_model_chunks - 1)

        def _process_fwd_buffer(step_id, tensor):
            if step_id < first_chunk_steps:
                if not self.is_pipeline_last_stage():
                    fwd_buffer_queue.put(tensor)
                if left_id <= step_id < right_id:
                    tensor = fwd_buffer_queue.get()
                else:
                    tensor = None
            else:
                if self.is_pipeline_last_stage():
                    tensor = None
            return tensor

        def _process_bwd_buffer(step_id, tensor):
            if step_id < first_chunk_steps:
                if not self.is_pipeline_first_stage():
                    bwd_buffer_queue.put(tensor)
                if left_id <= step_id < right_id:
                    tensor = bwd_buffer_queue.get()
                else:
                    tensor = None
            else:
                if self.is_pipeline_first_stage():
                    tensor = None
            return tensor

        per_stage_accumulate_steps = self.accumulate_steps // self.num_stages
        self._backward_step_count = -(
            first_chunk_steps
            + (per_stage_accumulate_steps - 2)
            * self.num_stages
            * self.num_model_chunks
        )

        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]

        micro_dataset = self._wrap_data(data)

        num_steps = self.accumulate_steps * self.num_model_chunks
        all_startup_steps = False
        if forward_only:
            # If only forward, since there is no backward during running, all steps are startup steps
            startup_steps = num_steps
        else:
            startup_steps = (self.num_stages - self.stage_id - 1) * 2
            startup_steps += (self.num_model_chunks - 1) * first_chunk_acc
            startup_steps = min(startup_steps, num_steps)

        steady_steps = num_steps - startup_steps

        self.set_virtual_pipeline_rank(0)
        self.input_tensors[0].append(
            self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                sync_recv=False,
                batch_p2p_comm=self._batch_p2p_comm,
            )
        )

        fwd_wait_handles = None
        bwd_wait_handles = None

        # run startup steps
        for micro_step in range(startup_steps):
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            output_tensor = self._forward_step_helper(micro_dataset, micro_step)

            # determine whether recv forward tensor or not
            next_virtual_pp_rank = self._get_virtual_pp_rank(
                micro_step + 1, forward=True
            )
            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if next_virtual_pp_rank == 0:
                    # next chunk is the first chunk, not need to pre recv an input tensor
                    recv_prev = False
            # last micro step, no next run
            if micro_step == (num_steps - 1):
                recv_prev = False

            # last stage shouldn't send tensor to downstream
            if self.is_pipeline_last_stage(ignore_virtual=True):
                output_tensor = _process_fwd_buffer(micro_step, output_tensor)

            if not self._overlap_p2p_comm:
                if (
                    micro_step == (startup_steps - 1)
                    and not forward_only
                    and not all_startup_steps
                ):
                    input_tensor_grad = None
                    recv_next = True
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    # the last startup step needs on four direction comm to set up for steady 1f1b
                    (
                        input_tensor,
                        output_tensor_grad,
                    ) = self._p2p_helper.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                    )
                    self.output_tensor_grads[self.num_model_chunks - 1].append(
                        output_tensor_grad
                    )
                else:
                    input_tensor = self._p2p_helper.send_forward_recv_forward(
                        output_tensor, recv_prev=recv_prev
                    )
                self.input_tensors[next_virtual_pp_rank].append(input_tensor)
            else:
                (
                    input_tensor,
                    fwd_wait_handles,
                ) = self._p2p_helper.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    batch_p2p_comm=self._batch_p2p_comm,
                    overlap_p2p_comm=True,
                )
                if (
                    micro_step == (startup_steps - 1)
                    and (not forward_only)
                    and steady_steps
                ):
                    input_tensor_grad = None
                    recv_next = True
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    (
                        output_tensor_grad,
                        bwd_wait_handles,
                    ) = self._p2p_helper.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        batch_p2p_comm=self._batch_p2p_comm,
                        overlap_p2p_comm=True,
                    )
                    self.output_tensor_grads[self.num_model_chunks - 1].append(
                        output_tensor_grad
                    )

                # append input_tensor no matter none or not
                self.input_tensors[next_virtual_pp_rank].append(input_tensor)

            self._release_output(output_tensor)

        # run 1f1b steady steps
        for micro_step in range(steady_steps):
            # forward
            forward_micro_step_id = micro_step + startup_steps

            if self._overlap_p2p_comm:
                if fwd_wait_handles is not None:
                    for req in fwd_wait_handles:
                        req.wait()

                self._release_output(output_tensor)
                output_tensor = self._forward_step_helper(
                    micro_dataset, forward_micro_step_id
                )

                forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id, forward=True
                )

                self.set_virtual_pipeline_rank(forward_virtual_pp_rank)
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    output_tensor = _process_fwd_buffer(
                        forward_micro_step_id, output_tensor
                    )

                # determine whether to recv input tensor from upstream
                recv_prev = True
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                        forward_micro_step_id + 1, forward=True
                    )
                    if next_forward_virtual_pp_rank == 0:
                        # next chunk is the first chunk, not need to pre recv an input tensor
                        recv_prev = False
                else:
                    next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                        forward_micro_step_id + 1, forward=True
                    )

                # last iteration doesn't need recv from upstream
                if micro_step == (steady_steps - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                (
                    input_tensor,
                    fwd_wait_handles,
                ) = self._p2p_helper.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    batch_p2p_comm=self._batch_p2p_comm,
                    overlap_p2p_comm=True,
                )

                if bwd_wait_handles is not None:
                    for req in bwd_wait_handles:
                        req.wait()

                # backward pass
                backward_micro_step_id = micro_step
                input_tensor_grad = self._backward_step_helper(
                    backward_micro_step_id
                )

                # first stage doesn't send grad to upstream
                backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id, forward=False
                )
                self.set_virtual_pipeline_rank(backward_virtual_pp_rank)
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor_grad = _process_bwd_buffer(
                        backward_micro_step_id, input_tensor_grad
                    )

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1,
                        forward=False,
                    )
                    if next_backward_virtual_pp_rank == (
                        self.num_model_chunks - 1
                    ):
                        # next chunk is the last chunk, not need to pre recv an output tensor grad
                        recv_next = False
                else:
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1, forward=False
                    )

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = self._p2p_helper.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    batch_p2p_comm=self._batch_p2p_comm,
                    overlap_p2p_comm=True,
                )
            else:
                output_tensor = self._forward_step_helper(
                    micro_dataset, forward_micro_step_id
                )

                # backward
                backward_micro_step_id = micro_step
                input_tensor_grad = self._backward_step_helper(
                    backward_micro_step_id
                )

                # four directions comm
                # send output tensor to downstream
                # send input tensor grad to upstream
                # recv input tensor from upstream
                # recv output tensor grad from downstream

                # last stage doesn't send rst to downstream
                forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id, forward=True
                )
                self.set_virtual_pipeline_rank(forward_virtual_pp_rank)

                if self.is_pipeline_last_stage(ignore_virtual=True):
                    output_tensor = _process_fwd_buffer(
                        forward_micro_step_id, output_tensor
                    )

                # first stage doesn't send grad to upstream
                backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id, forward=False
                )
                self.set_virtual_pipeline_rank(backward_virtual_pp_rank)

                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor_grad = _process_bwd_buffer(
                        backward_micro_step_id, input_tensor_grad
                    )

                # determine whether to recv input tensor from upstream
                recv_prev = True
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                        forward_micro_step_id + 1, forward=True
                    )
                    if next_forward_virtual_pp_rank == 0:
                        # next chunk is the first chunk, not need to pre recv an input tensor
                        recv_prev = False

                else:
                    next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                        forward_micro_step_id + 1, forward=True
                    )

                # last iteration doesn't need recv from upstream
                if micro_step == (steady_steps - 1):
                    recv_prev = False

                # determine whether to recv grad from downstream
                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1,
                        forward=False,
                    )
                    if next_backward_virtual_pp_rank == (
                        self.num_model_chunks - 1
                    ):
                        # next chunk is the last chunk, not need to pre recv an output tensor grad
                        recv_next = False
                else:
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1, forward=False
                    )

                (
                    input_tensor,
                    output_tensor_grad,
                ) = self._p2p_helper.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                )

            self._release_output(output_tensor)

            if recv_prev:
                self.input_tensors[next_forward_virtual_pp_rank].append(
                    input_tensor
                )
            if recv_next:
                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    output_tensor_grad
                )

        assert fwd_buffer_queue.empty(), "forward buffer should be empty"

        self._release_output(output_tensor)

        # remaining backward steps
        if not forward_only:
            if self._overlap_p2p_comm and bwd_wait_handles is not None:
                for wait_handles in bwd_wait_handles:
                    wait_handles.wait()
            if all_startup_steps:
                self.output_tensor_grads[self.num_model_chunks - 1].append(
                    self._p2p_helper.recv_backward(
                        self.is_pipeline_last_stage(),
                        batch_p2p_comm=self._batch_p2p_comm,
                    )
                )

            for micro_step in range(steady_steps, num_steps):
                # cooldown loop
                input_tensor_grad = self._backward_step_helper(micro_step)
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step + 1, forward=False
                )

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_virtual_pp_rank == (
                        self.num_model_chunks - 1
                    ):
                        recv_next = False

                if micro_step == (num_steps - 1):
                    recv_next = False

                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor_grad = _process_bwd_buffer(
                        micro_step, input_tensor_grad
                    )

                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    self._p2p_helper.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        batch_p2p_comm=self._batch_p2p_comm,
                    )
                )

            self._sync_overlap_grads()

            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").start()
            self._layers.allreduce_shared_weight_gradients()
            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").stop()

        assert bwd_buffer_queue.empty(), "backward buffer should be empty"

        if compute_loss:
            # return loss if compute loss
            if self._enable_timer:
                self.timers("broadcast_final_loss").start()
            with paddle.amp.auto_cast(enable=False):
                train_loss = self._broadcast_final_loss()
            if self._enable_timer:
                self.timers("broadcast_final_loss").stop()
        else:
            # else just return all intermediate output tensor for all micro steps
            train_loss = self.output_tensors

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()
        return train_loss

    def train_batch(self, data, optimizer, lr_scheduler=None, scaler=None):
        data = self._prepare_training(data, optimizer, lr_scheduler)
        # interleave scheduler for pipeline parallel
        train_loss = self.forward_backward_pipeline(data, scaler)

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(self, data, compute_loss=False):
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        self._compute_loss = compute_loss

        return self.forward_backward_pipeline(data, None, forward_only=True)


class PipelineParallelWithInterleaveFthenB(PipelineParallelWithInterleave):
    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)

    def _check_sanity(self):
        assert (
            framework.in_dygraph_mode()
        ), "virtual pipeline stage with interleave only support eager dygraph mode"

        assert (
            self.num_stages > 2
        ), "virtual pipeline must run under pp degree > 2"

    def _get_virtual_pp_rank(self, micro_step, forward):

        virtual_pp_stage = micro_step % (
            self.accumulate_steps * self.num_model_chunks
        )
        virtual_pp_stage = virtual_pp_stage // self.accumulate_steps
        if not forward:
            virtual_pp_stage = self.num_model_chunks - virtual_pp_stage - 1

        return virtual_pp_stage

    def _overlap_comm_grads(self):
        if not self._comm_overlap:
            return
        self._backward_step_count += 1
        sync_step = self._backward_step_count - self.stage_id

        if sync_step > 0 and sync_step % self.accumulate_steps == 0:
            chunk_idx = self._virtual_pp_world_size - (
                sync_step // self.accumulate_steps
            )
            old_value = set_nccl_channels(nccl_channel_num_for_overlap_comm)
            for buffer in self._chunk_2_comm_buffers[chunk_idx]:
                buffer.comm_grads()
            set_nccl_channels(old_value)

        if self.stage_id == 0:
            return

        if (
            self._backward_step_count
            == self.accumulate_steps * self._virtual_pp_world_size
        ):
            old_value = set_nccl_channels(nccl_channel_num_for_overlap_comm)
            for buffer in self._chunk_2_comm_buffers[0]:
                buffer.comm_grads()
            set_nccl_channels(old_value)

    def _sync_overlap_grads(self):
        if not self._comm_overlap:
            return

        expected_count = self.accumulate_steps * self._virtual_pp_world_size
        assert self._backward_step_count == expected_count, (
            f"backward step count should be equal to accumulate steps * virtual pp world size, "
            f"but got {self._backward_step_count}, expected result is {expected_count}"
        )

        for buffers in self._chunk_2_comm_buffers.values():
            for buffer in buffers:
                buffer.scale_and_split_grads()

    def forward_backward_pipeline(
        self, data, scaler, forward_only=False, compute_loss=True
    ):
        if not compute_loss:
            assert (
                not forward_only
            ), "compute_loss can only be set to False when forward_only is set to True"

        # NOTE(shenliang03): Due to ring_exchange for pipeline with interleave, cache should be enabled
        assert (
            self._using_cache
        ), "cache should be enabled for pipeline with interleave"

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only

        assert (
            self.accumulate_steps == self.num_stages
            or self.accumulate_steps % self.num_stages != 0
        ), "accumulate_steps({}) and num_stages({}) should be a multiple or accumulate_steps % num_stages == 0".format(
            self.accumulate_steps, self.num_stages
        )

        self._backward_step_count = 0
        skip_steps = self.accumulate_steps - self.num_stages
        send_recv_buffer_queue = queue.Queue()

        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]

        micro_dataset = self._wrap_data(data)
        num_steps = self.accumulate_steps * self.num_model_chunks

        self.set_virtual_pipeline_rank(0)
        self.input_tensors[0].append(
            self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(), sync_recv=False
            )
        )

        # run startup steps
        for micro_step in range(num_steps):
            output_tensor = self._forward_step_helper(micro_dataset, micro_step)
            # determine whether recv forward tensor or not
            next_virtual_pp_rank = self._get_virtual_pp_rank(
                micro_step + 1, forward=True
            )

            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if next_virtual_pp_rank == 0:
                    # next chunk is the first chunk, not need to pre recv an input tensor
                    recv_prev = False

            # last micro step, no next run
            if micro_step == (num_steps - 1):
                recv_prev = False

            if self.is_pipeline_last_stage(ignore_virtual=True):
                # last stage skip send/recv
                if not self.is_pipeline_last_stage():
                    send_recv_buffer_queue.put(output_tensor)

                if micro_step < skip_steps or (
                    self.is_pipeline_last_stage()
                    and micro_step % self.accumulate_steps >= skip_steps
                ):
                    output_tensor = None
                else:
                    output_tensor = send_recv_buffer_queue.get()

            input_tensor = self._p2p_helper.send_forward_recv_forward(
                output_tensor, recv_prev=recv_prev
            )
            self.input_tensors[next_virtual_pp_rank].append(input_tensor)

            self._release_output(output_tensor)

        assert (
            send_recv_buffer_queue.empty()
        ), "send_recv buffer should be empty"

        # remaining backward steps
        if not forward_only:

            self.output_tensor_grads[self.num_model_chunks - 1].append(
                self._p2p_helper.recv_backward(
                    self.is_pipeline_last_stage(), sync_recv=False
                )
            )

            for micro_step in range(num_steps):
                # cooldown loop
                input_tensor_grad = self._backward_step_helper(micro_step)
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step + 1, forward=False
                )

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_virtual_pp_rank == (
                        self.num_model_chunks - 1
                    ):
                        recv_next = False

                if micro_step == (num_steps - 1):
                    recv_next = False

                if self.is_pipeline_first_stage(ignore_virtual=True):
                    if not self.is_pipeline_first_stage():
                        send_recv_buffer_queue.put(input_tensor_grad)

                    if micro_step < skip_steps or (
                        self.is_pipeline_first_stage()
                        and micro_step % self.accumulate_steps >= skip_steps
                    ):
                        input_tensor_grad = None
                    else:
                        input_tensor_grad = send_recv_buffer_queue.get()

                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    self._p2p_helper.send_backward_recv_backward(
                        input_tensor_grad, recv_next=recv_next
                    )
                )

            assert (
                send_recv_buffer_queue.empty()
            ), "send_recv buffer should be empty"

            self._sync_overlap_grads()

            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").start()
            self._layers.allreduce_shared_weight_gradients()
            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").stop()

        if compute_loss:
            # return loss if compute loss
            if self._enable_timer:
                self.timers("broadcast_final_loss").start()
            with paddle.amp.auto_cast(enable=False):
                train_loss = self._broadcast_final_loss()
            if self._enable_timer:
                self.timers("broadcast_final_loss").stop()
        else:
            # else just return all intermediate output tensor for all micro steps
            train_loss = self.output_tensors

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()
        return train_loss
