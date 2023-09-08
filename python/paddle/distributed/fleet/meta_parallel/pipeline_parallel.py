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
import sys
import time
import warnings
from collections import defaultdict

import paddle
from paddle import framework

from ..meta_optimizers.dygraph_optimizer import HybridParallelOptimizer
from ..utils import timer_helper as timer
from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters,
    broadcast_mp_parameters,
    broadcast_sharding_parameters,
)
from ..utils.log_util import logger
from .meta_parallel_base import MetaParallelBase
from .parallel_layers.pp_layers import PipelineLayer

_use_four_directions = os.environ.get(
    'PADDLE_USE_FOUR_DIRECTIONS_P2P', paddle.base.core.is_compiled_with_xpu()
)
if _use_four_directions:
    from .pp_utils import four_directions_p2p_communication as p2p
else:
    from .pp_utils import p2p_communication as p2p

from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
)

__all__ = []

g_shard_use_reduce = int(os.environ.get("FLAGS_shard_use_reduce", 1))


# assume only the first stage and last stage need data, and data consumption is ordred
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
        if self._index >= self._acc_steps:
            raise StopIteration
        assert self._is_first_stage or self._is_last_stage
        micro_batch_data = self._load_micro_batch(self._index)
        self._index += 1
        return micro_batch_data

    def _load_micro_batch(self, micro_step):
        inputs = self._data

        if self._is_first_stage or self._is_last_stage:
            assert len(inputs) == 2, "length of input should be 2"
            data = self._load_micro_batch_impl(inputs[0], micro_step)
            label = self._load_micro_batch_impl(inputs[1], micro_step)
            return (data, label)
        else:
            return (None, None)

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
                    output.append(data[micro_step].detach())
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
        self.global_rank = self._hcg.get_global_rank()
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

        self._profiling = self._strategy.hybrid_configs["pp_configs"].profiling
        self._records = []
        self._record_format = (
            '"name": "{}{}", "cat": "pipeline timeline", "ph": {}, "pid": 0, "tid": '
            + str(self.stage_id + 1)
            + ', "ts": {}, "cname": "{}"'
        )
        self._forward_color = "thread_state_running"  # RGB: 126, 200, 148
        self._backward_color = "rail_idle"  # RGB: 238, 142, 0
        if self._profiling:
            logger.info(
                "If enable pp profiling, the max training steps should be restricted "
                "to a reasonable value (such as 5) to avoid generating large profile files. "
                "The profiler will generate a profile file 'profile_record_tmp_file_for_rank_*' "
                "for each rank. Users should gather all profile files for one entire pipeline "
                "to one node (rank 0 is recommended) to get the full view of the pipeline profile. "
                "[DONT CHANGE THE NAME OF THE PROFILE FILES!]. "
                "Then get the profile parser from this url: "
                "https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/fleet/meta_parallel/pp_utils/profiler_helper.py "
                "and save the script to the same directory of all profile files."
                "Parse those files by this command: `python profiler_helper.py`. "
                "After parsing, a new file 'pipeline_profile.json' will be generated. "
                "Users can inspect this file by chrome://tracing website."
            )

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
            self._using_cache,
            self._enable_partial_send_recv,
            self._enable_timer,
        )

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

    def bw_hook_func(self, buffer, param):
        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            buffer.add_grad(param)

        return fused_allreduce

    def register_allreduce_overlap_hook(
        self, model, comm_group, acc_steps, dp, group_size=128 * 1024 * 1024
    ):
        if model.get_num_virtual_stages() > 1:
            models = model.get_model_chunks()
        else:
            models = [model]

        if not dp:
            assert hasattr(self, "optimizer")
            assert hasattr(self.optimizer, "_param2rank")
            _param2rank = self.optimizer._param2rank
        # Note: after sharding change to reduce operation, here need to be cleared
        act = (
            HOOK_ACTION.ALL_REDUCE
            if (dp or not g_shard_use_reduce)
            else HOOK_ACTION.REDUCE
        )

        for chunk_idx, model in enumerate(models):
            # For virtual pipeline. Will separate parameters in different chunk into
            # different groups to get the best performance.

            fused_parameter_group = {}
            parameter_list = [
                p for p in model.parameters() if not p.stop_gradient
            ]
            if len(parameter_list) < 1:
                return

            if dp:
                fused_parameter_group[-1] = parameter_list
            else:
                # Sort parameters for sharding, since they have different dst rank
                for p in parameter_list:
                    assert p.name in _param2rank
                    dst_rank = _param2rank[p.name]
                    if dst_rank in fused_parameter_group:
                        fused_parameter_group[dst_rank].append(p)
                    else:
                        fused_parameter_group[dst_rank] = [p]

            for dst in fused_parameter_group:
                parameter_list = fused_parameter_group[dst]
                if act != HOOK_ACTION.ALL_REDUCE:
                    # parse the relative dst rank to absolute dst rank for sharding
                    dst = comm_group.ranks[dst]
                else:
                    dst = -1
                var_groups = assign_group_by_size(parameter_list, group_size)
                for group_idx, parameters in var_groups.items():
                    buffer = FusedCommBuffer(
                        group_idx, parameters, comm_group, acc_steps, act, dst
                    )
                    self._chunk_2_comm_buffers[chunk_idx].append(buffer)
                    for param in parameters:
                        param._register_backward_hook(
                            self.bw_hook_func(buffer, param)
                        )

    def timer_printer(self):
        if not self._enable_timer:
            return
        all_flag_names = self.timers.timers.keys()
        self.timers.log(all_flag_names)

    def _record_stamp(self, name, step, phase, color):
        if self._profiling:
            paddle.device.synchronize()
            self._records.append(
                '{'
                + self._record_format.format(
                    name,
                    step,
                    phase,
                    int(time.time() * 1000),
                    color,
                )
                + '}'
            )

    def _flush_records(self):
        if self._profiling:
            with open(
                f'./profile_record_tmp_file_for_rank_{self.global_rank}',
                'a+',
            ) as f:
                for record in self._records:
                    f.write(record + '\n')
            self._records = []

    def forward_backward_pipeline(
        self, data, scaler=None, static_scheduler=False
    ):
        # use the 1f1b scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py

        if static_scheduler:
            assert (
                not self._profiling
            ), "While _profiling, static scheduler is not available"
            if data is not None:
                warnings.warn(
                    "Static scheduler run won't real run the model, but data has been provided"
                )
            logger.info(
                "enable static_scheduler will return the pp schedule instead of the loss"
            )
            schedule = ""

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
            if static_scheduler:
                schedule += f"f{step_id};"
                logger.info(f"forward step for micro step {step_id}")
                continue
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

            self._record_stamp("F", step_id, '"B"', self._forward_color)
            output_tensor = self._forward_step(input_tensor, micro_dataset)
            self._record_stamp("F", step_id, '"E"', self._forward_color)
            p2p.send_forward(output_tensor, self.is_pipeline_last_stage())

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)

        if steady_steps > 0 and not static_scheduler:
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

        for i in range(steady_steps):
            if static_scheduler:
                schedule += f"f{startup_steps + i};"
                schedule += f"b{i};"
                logger.info(f"forward step for micro step {startup_steps + i}")
                logger.info(f"backward step for micro step {i}")
                continue
            last_iter = i == (steady_steps - 1)

            self._record_stamp(
                "F", startup_steps + i, '"B"', self._forward_color
            )
            output_tensor = self._forward_step(input_tensor, micro_dataset)
            self._record_stamp(
                "F", startup_steps + i, '"E"', self._forward_color
            )

            output_tensor_grad = p2p.send_forward_recv_backward(
                output_tensor, self.is_pipeline_last_stage()
            )

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not self.is_pipeline_last_stage():
                self._release_output(output_tensor)

            input_tensor, output_tensor = input_buffers.pop(
                0
            ), output_buffers.pop(0)

            self._record_stamp("B", i, '"B"', self._backward_color)
            input_tensor_grad = self._backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )
            self._record_stamp("B", i, '"E"', self._backward_color)

            if last_iter:
                input_tensor = None
                p2p.send_backward(
                    input_tensor_grad, self.is_pipeline_first_stage()
                )
            else:
                input_tensor = p2p.send_backward_recv_forward(
                    input_tensor_grad, self.is_pipeline_first_stage()
                )

        for i in range(startup_steps):
            if static_scheduler:
                schedule += f"b{steady_steps + i};"
                logger.info(f"backward step for micro step {steady_steps + i}")
                continue
            input_tensor = input_buffers.pop(0)
            output_tensor = output_buffers.pop(0)

            output_tensor_grad = p2p.recv_backward(
                self.is_pipeline_last_stage()
            )

            self._record_stamp(
                "B", steady_steps + i, '"B"', self._backward_color
            )
            input_tensor_grad = self._backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )
            self._record_stamp(
                "B", steady_steps + i, '"E"', self._backward_color
            )
            p2p.send_backward(input_tensor_grad, self.is_pipeline_first_stage())

        if static_scheduler:
            return schedule

        self._flush_records()

        if self._comm_overlap:
            assert (
                len(self._chunk_2_comm_buffers) > 0
            ), "comm buffers should be created"
            for _, buffers in self._chunk_2_comm_buffers.items():
                for buffer in buffers:
                    buffer.scale_grads()

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
        self.timer_printer()
        return train_loss

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

        if self._sharding_comm_overlap and len(self._chunk_2_comm_buffers) == 0:
            self.register_allreduce_overlap_hook(
                self._layers, self.sharding_group, self.accumulate_steps, False
            )

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

        # convert to micro dataset
        micro_dataset = self._wrap_data(data)

        for step_id in range(startup_steps):
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

            output_tensor = self._forward_step(input_tensor, micro_dataset)
            p2p.send_forward(output_tensor, self.is_pipeline_last_stage())

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

        if steady_steps > 0:
            input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

        for i in range(steady_steps):
            last_iter = i == (steady_steps - 1)

            output_tensor = self._forward_step(input_tensor, micro_dataset)
            p2p.send_forward(output_tensor, self.is_pipeline_last_stage())

            input_buffers.append(input_tensor)
            output_buffers.append(output_tensor)

            if not last_iter:
                input_tensor = p2p.recv_forward(self.is_pipeline_first_stage())

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

    def _check_micro_batch_data_valid(self, micro_batch_data):
        if isinstance(micro_batch_data, (tuple, list)):
            for data in micro_batch_data:
                self._check_micro_batch_data_valid(data)
        elif micro_batch_data is not None:
            assert isinstance(micro_batch_data, paddle.Tensor)

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

    def get_static_scheduler(self):
        return self.forward_backward_pipeline(data=None, static_scheduler=True)


class PipelineParallelWithInterleave(PipelineParallel):
    # pipeline parallel with interleave scheduler

    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        self._record_format = (
            '"name": "{}{}_VP{}", "cat": "virtual pipeline timeline", "ph": {}, "pid": 0, "tid": '
            + str(self.stage_id + 1)
            + ', "ts": {}, "cname": "{}"'
        )
        self._forward_colors = [
            "thread_state_running",  # RGB: 126, 200, 148
            "thread_state_unknown",  # RGB: 199, 155, 125
        ]
        self._backward_colors = [
            "rail_load",  # RGB: 13, 168, 97
            "rail_idle",  # RGB: 238, 142, 0
        ]
        # Structures to record the micro step for each layer chunk
        self._forward_micro_step_counter = {}
        self._backward_micro_step_counter = {}

        assert layers.get_num_virtual_stages() > 1
        assert (
            self.num_stages > 2
        ), "virtual pipeline must run under pp degree > 2"
        assert (
            framework.in_dynamic_mode()
        ), "virtual pipeline stage with interleave only support eager dygraph mode"
        assert (
            self.accumulate_steps % self.num_stages == 0
        ), "accumulate_steps should be evenly divisible by num_stages for pipeline with interleave"
        # setup for interleave scheduler
        self.num_model_chunks = layers.get_num_virtual_stages()
        self.model_chunks = layers.get_model_chunks()
        assert self.model_chunks is not None
        assert len(self.model_chunks) == self.num_model_chunks
        self._virtual_pp_world_size = self.num_model_chunks
        self._virtual_pp_rank = 0
        self._reset_counter()

    def _reset_counter(self):
        for i in range(self.num_model_chunks):
            self._forward_micro_step_counter[i] = 0
            self._backward_micro_step_counter[i] = 0

    def _record_stamp(self, name, step, phase, forward=True):
        if self._profiling:
            paddle.device.synchronize()
            virtual_pp_rank = self._get_virtual_pp_rank(step, forward=forward)
            color_idx = virtual_pp_rank % 2
            # Get the profile color and micro step for current layer chunk
            if forward:
                color = self._forward_colors[color_idx]
                micro_step = self._forward_micro_step_counter[virtual_pp_rank]
                if phase == '"E"':
                    self._forward_micro_step_counter[virtual_pp_rank] += 1
            else:
                color = self._backward_colors[color_idx]
                micro_step = self._backward_micro_step_counter[virtual_pp_rank]
                if phase == '"E"':
                    self._backward_micro_step_counter[virtual_pp_rank] += 1
            self._records.append(
                '{'
                + self._record_format.format(
                    name,
                    micro_step,
                    virtual_pp_rank,
                    phase,
                    int(time.time() * 1000),
                    color,
                )
                + '}'
            )

    def _flush_records(self):
        if self._profiling:
            with open(
                f'./profile_record_tmp_file_for_rank_{self.global_rank}',
                'a+',
            ) as f:
                for record in self._records:
                    f.write(record + '\n')
            self._records = []
            self._reset_counter()

    def _get_virtual_pp_rank(self, micro_step, forward):
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
        assert len(self.input_tensors[virtual_pp_rank]) == (
            len(self.output_tensors[virtual_pp_rank]) + 1
        )
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
                for buffer in self._chunk_2_comm_buffers[chunk_idx]:
                    buffer.comm_grads()

            if self.stage_id != 0:
                if (
                    self._backward_step_count
                    == self.num_stages * self.num_model_chunks
                ):
                    for buffer in self._chunk_2_comm_buffers[0]:
                        buffer.comm_grads()

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
                    buffer.scale_grads()

    def _backward_step_helper(self, micro_step):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=False)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        assert hasattr(self, 'output_tensor_grads')

        assert (
            len(self.output_tensor_grads[virtual_pp_rank]) == 1
        ), f"output_tensor_grads is empty for virtual_pp_rank {virtual_pp_rank}"

        assert len(self.input_tensors[virtual_pp_rank]) > 0
        assert len(self.output_tensors[virtual_pp_rank]) > 0

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
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
        static_scheduler=False,
    ):
        # use interleave scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py
        if not compute_loss:
            assert (
                not forward_only
            ), "compute_loss can only be set to False when forward_only is set to True"

        if static_scheduler:
            assert (
                not forward_only
            ), "static_scheduler only for training not for eval"
            assert (
                not self._profiling
            ), "While _profiling, static scheduler is not available"
            if data is not None:
                warnings.warn(
                    "Static scheduler run won't real run the model, but data has been provided"
                )
            logger.info(
                "enable static_scheduler will return the pp schedule instead of the loss"
            )
            schedule = ""

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only

        # store the number of backward steps
        assert (
            self.accumulate_steps % self.num_stages == 0
        ), "accumulate_steps({}) should be evenly divisible by num_stages({}) for pipeline with interleave".format(
            self.accumulate_steps, self.num_stages
        )
        per_stage_accumulate_steps = self.accumulate_steps // self.num_stages
        self._backward_step_count = (
            -(per_stage_accumulate_steps - 1)
            * self.num_stages
            * self.num_model_chunks
        )

        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]

        micro_dataset = self._wrap_data(data)

        num_steps = self.accumulate_steps * self.num_model_chunks
        if forward_only:
            # If only forward, since there is no backward during running, all steps are startup steps
            startup_steps = num_steps
        else:
            # actually startup_steps is calculated from two number:
            # first_forward_cross_to_end = (self.num_stages - self.stage_id - 1) + (self.num_model_chunks - 1) * self.num_stages
            # end_to_first_backward_cross = (self.num_stages - self.stage_id - 1)
            # startup_steps = first_forward_cross_to_end + end_to_first_backward_cross
            startup_steps = (self.num_stages - self.stage_id - 1) * 2
            startup_steps += (self.num_model_chunks - 1) * self.num_stages
            startup_steps = min(startup_steps, num_steps)

        steady_steps = num_steps - startup_steps

        self.set_virtual_pipeline_rank(0)
        if not static_scheduler:
            self.input_tensors[0].append(
                p2p.recv_forward(
                    self.is_pipeline_first_stage(), sync_recv=False
                )
            )

        # run startup steps
        for micro_step in range(startup_steps):
            if static_scheduler:
                virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step, forward=True
                )
                real_micro_step = self._forward_micro_step_counter[
                    virtual_pp_rank
                ]
                self._forward_micro_step_counter[virtual_pp_rank] += 1
                schedule += f"f{real_micro_step}_vp{virtual_pp_rank};"
                logger.info(
                    f"forward step for {real_micro_step} with virtual pp rank {virtual_pp_rank}"
                )
                continue

            self._record_stamp("F", micro_step, '"B"', forward=True)
            output_tensor = self._forward_step_helper(micro_dataset, micro_step)
            self._record_stamp("F", micro_step, '"E"', forward=True)

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
            if self.is_pipeline_last_stage():
                output_tensor = None

            if micro_step == (startup_steps - 1) and not forward_only:
                input_tensor_grad = None
                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                # the last startup step needs on four direction comm to set up for steady 1f1b
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                )
                # output_tensor_grad is not none if recv_next
                # append output_tensor_grad no matter none or not
                self.output_tensor_grads[self.num_model_chunks - 1].append(
                    output_tensor_grad
                )
            else:
                input_tensor = p2p.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev
                )
            # append input_tensor no matter none or not
            self.input_tensors[next_virtual_pp_rank].append(input_tensor)

            self._release_output(output_tensor)

        # run 1f1b steady steps
        for micro_step in range(steady_steps):
            if static_scheduler:
                forward_micro_step_id = micro_step + startup_steps
                forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id, forward=True
                )
                backward_micro_step_id = micro_step
                backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    backward_micro_step_id, forward=False
                )
                real_forward_micro_step = self._forward_micro_step_counter[
                    forward_virtual_pp_rank
                ]
                self._forward_micro_step_counter[forward_virtual_pp_rank] += 1
                real_backward_micro_step = self._backward_micro_step_counter[
                    backward_virtual_pp_rank
                ]
                self._backward_micro_step_counter[backward_virtual_pp_rank] += 1
                schedule += (
                    f"f{real_forward_micro_step}_vp{forward_virtual_pp_rank};"
                )
                schedule += (
                    f"b{real_backward_micro_step}_vp{backward_virtual_pp_rank};"
                )
                logger.info(
                    f"forward step for {real_forward_micro_step} with virtual pp rank {forward_virtual_pp_rank}"
                )
                logger.info(
                    f"backward step for {real_backward_micro_step} with virtual pp rank {backward_virtual_pp_rank}"
                )
                continue
            # forward
            forward_micro_step_id = micro_step + startup_steps
            self._record_stamp("F", forward_micro_step_id, '"B"', forward=True)
            output_tensor = self._forward_step_helper(
                micro_dataset, forward_micro_step_id
            )
            self._record_stamp("F", forward_micro_step_id, '"E"', forward=True)

            # backward
            backward_micro_step_id = micro_step
            self._record_stamp(
                "B", backward_micro_step_id, '"B"', forward=False
            )
            input_tensor_grad = self._backward_step_helper(
                backward_micro_step_id
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"E"', forward=False
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
            if self.is_pipeline_last_stage():
                output_tensor = None

            # first stage doesn't send grad to upstream
            backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id, forward=False
            )
            self.set_virtual_pipeline_rank(backward_virtual_pp_rank)
            if self.is_pipeline_first_stage():
                input_tensor_grad = None

            # determine whether to recv input tensor from upstream
            recv_prev = True
            next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                forward_micro_step_id + 1, forward=True
            )
            if self.is_pipeline_first_stage(ignore_virtual=True) and (
                next_forward_virtual_pp_rank == 0
            ):
                # first pp stage and first virtual stage
                recv_prev = False

            # last iteration doesn't need recv from upstream
            if micro_step == (steady_steps - 1):
                recv_prev = False

            # determine whether to recv grad from downstream
            recv_next = True
            next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id + 1, forward=False
            )
            if self.is_pipeline_last_stage(ignore_virtual=True) and (
                next_backward_virtual_pp_rank == (self.num_model_chunks - 1)
            ):
                # last pp stage and last virtual stage
                recv_next = False

            (
                input_tensor,
                output_tensor_grad,
            ) = p2p.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
            )
            # append input_tensor no matter none or not
            self.input_tensors[next_forward_virtual_pp_rank].append(
                input_tensor
            )
            # append output_tensor_grad no matter none or not
            self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                output_tensor_grad
            )
            self._release_output(output_tensor)

        if not static_scheduler:
            self._release_output(output_tensor)

        # remaining backward steps
        if not forward_only:
            for micro_step in range(steady_steps, num_steps):
                if static_scheduler:
                    virtual_pp_rank = self._get_virtual_pp_rank(
                        micro_step, forward=False
                    )
                    real_micro_step = self._backward_micro_step_counter[
                        virtual_pp_rank
                    ]
                    self._backward_micro_step_counter[virtual_pp_rank] += 1
                    schedule += f"b{real_micro_step}_vp{virtual_pp_rank};"
                    logger.info(
                        f"backward step for {real_micro_step} with virtual pp rank {virtual_pp_rank}"
                    )
                    continue
                # cooldown loop
                self._record_stamp("B", micro_step, '"B"', forward=False)
                input_tensor_grad = self._backward_step_helper(micro_step)
                self._record_stamp("B", micro_step, '"E"', forward=False)
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
                # append output_tensor_grad no matter none or not
                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    p2p.send_backward_recv_backward(
                        input_tensor_grad, recv_next=recv_next
                    )
                )

            self._sync_overlap_grads()

            if static_scheduler:
                self._reset_counter()
                return schedule

            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").start()
            self._layers.allreduce_shared_weight_gradients()
            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").stop()

        self._flush_records()

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

    def get_static_scheduler(self):
        return self.forward_backward_pipeline(
            data=None, scaler=None, static_scheduler=True
        )
