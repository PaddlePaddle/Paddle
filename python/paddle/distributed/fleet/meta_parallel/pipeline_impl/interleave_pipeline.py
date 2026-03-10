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
"""Pipeline Parallel with Interleave Implementation.

This module contains the PipelineParallelWithInterleave class which implements
pipeline parallelism with virtual pipeline stages (interleave mode).
"""

from __future__ import annotations

import os
import time

import paddle
from paddle import framework

from ..utils.log_util import logger
from .base_pipeline import PipelineParallel
from .pipeline_hooks import PipelineHook
from .pp_utils.utils import dict_to_tuple_helper, tuple_to_dict_helper

g_profile_pipeline_details_steps = int(
    os.getenv("FLAGS_profile_pipeline_details_steps", "0")
)


class PipelineParallelWithInterleave(PipelineParallel):
    # pipeline parallel with interleave scheduler

    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        self.overlap_schedule_mode = (
            hasattr(type(self._layers), "overlapped_forward_backward")
            and self._strategy.hybrid_configs[
                "pp_configs"
            ].forward_backward_overlap_scheduler
        )

        if self.overlap_schedule_mode:
            assert not self._profiling, (
                "Profiling is not compatible with overlap_schedule_mode."
            )
        logger.info(f"Using {self._get_scheduler_name()}")

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

        # setup for interleave scheduler
        self._check_sanity()
        self.num_model_chunks = layers.get_num_virtual_stages()
        self.model_chunks = layers.get_model_chunks()
        assert self.model_chunks is not None
        assert len(self.model_chunks) == self.num_model_chunks
        self._virtual_pp_world_size = self.num_model_chunks
        self._virtual_pp_rank = 0
        self._reset_counter()
        self._best_unbalanced_scheduler = self._strategy.hybrid_configs[
            "pp_configs"
        ].best_unbalanced_scheduler
        if self._best_unbalanced_scheduler:
            assert not self._comm_overlap, (
                "pp best unbalaced scheduler can not run together with dp/sharding overlap"
            )

        self._enable_offload_queue = self._strategy.hybrid_configs[
            "pp_configs"
        ].enable_offload_queue

        # reinit user hook since now we have virtual stages
        self._init_user_hooks()

    def _get_scheduler_name(self):
        return f"PipelineParallelWithInterleave with overlapping forward backward={self.overlap_schedule_mode}, overlap p2p comm={self._overlap_p2p_comm}"

    def _init_user_bubble_hooks(self):
        # initialize bubble hooks
        self.bubble_hooks = PipelineHook()
        self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def _check_sanity(self):
        assert framework.in_dynamic_mode(), (
            "virtual pipeline stage with interleave only support eager dygraph mode"
        )

        assert self.num_stages > 2, (
            "virtual pipeline must run under pp degree > 2"
        )

        assert self.accumulate_steps >= 2 * self.num_stages, (
            f"accumulate_steps({self.accumulate_steps}) should be greater than or equal to 2 * num_stages({self.num_stages}) for pipeline with interleave"
        )

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
                f.writelines(record + '\n' for record in self._records)
            self._records = []
            self._reset_counter()

    def _get_virtual_pp_rank(self, micro_step, forward):
        first_chunk_acc = (
            self.accumulate_steps % self.num_stages + self.num_stages
        )
        first_chunk_steps = first_chunk_acc * self.num_model_chunks
        if self._best_unbalanced_scheduler:
            num_group_last_chunk_forward = (
                (micro_step - first_chunk_acc) // self.num_stages
            ) // self.num_model_chunks
            misplace_start = (
                first_chunk_acc
                + self.num_model_chunks
                * self.num_stages
                * num_group_last_chunk_forward
            )
            misplace_end = (
                self.accumulate_steps % self.num_stages
                + num_group_last_chunk_forward * self.num_stages
            ) * self.num_model_chunks + self.num_stages
            forward_virtual_pp_stage = (
                (micro_step - first_chunk_acc) // self.num_stages
            ) % self.num_model_chunks

        if micro_step < first_chunk_steps:
            virtual_pp_stage = micro_step // first_chunk_acc
            if not forward and self._best_unbalanced_scheduler:
                if (
                    micro_step
                    >= first_chunk_acc
                    + (self.num_model_chunks - 1) * self.num_stages
                ):
                    if forward_virtual_pp_stage == self.num_model_chunks - 1:
                        virtual_pp_stage = 0
                    elif (
                        micro_step >= misplace_start
                        and micro_step < misplace_end
                    ):
                        virtual_pp_stage = (
                            micro_step - self.num_stages
                        ) // first_chunk_acc
        else:
            origin_micro_step = micro_step
            micro_step -= first_chunk_steps
            virtual_pp_stage = micro_step % (
                self.num_stages * self.num_model_chunks
            )
            virtual_pp_stage = virtual_pp_stage // self.num_stages
            if not forward and self._best_unbalanced_scheduler:
                total_num_forward_step_from_steady = (
                    first_chunk_acc
                    + (self.accumulate_steps - first_chunk_acc)
                    * self.num_model_chunks
                )
                if (
                    origin_micro_step <= total_num_forward_step_from_steady
                    and forward_virtual_pp_stage == self.num_model_chunks - 1
                ):
                    virtual_pp_stage = 0
                elif (
                    misplace_start <= total_num_forward_step_from_steady
                    and origin_micro_step >= misplace_start
                    and origin_micro_step < misplace_end
                ):
                    if origin_micro_step < first_chunk_steps + self.num_stages:
                        virtual_pp_stage = (
                            origin_micro_step - self.num_stages
                        ) // first_chunk_acc
                    else:
                        virtual_pp_stage = (micro_step - self.num_stages) % (
                            self.num_stages * self.num_model_chunks
                        )
                        virtual_pp_stage = virtual_pp_stage // self.num_stages

        if not forward:
            virtual_pp_stage = self.num_model_chunks - virtual_pp_stage - 1

        return virtual_pp_stage

    def _get_forward_input(self, virtual_pp_rank):
        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        if not self._forward_only:
            assert hasattr(self, 'output_tensor_grads')
            assert len(self.input_tensors[virtual_pp_rank]) == (
                len(self.output_tensors[virtual_pp_rank]) + 1
            )
            input_tensor = self.input_tensors[virtual_pp_rank][-1]
        else:
            input_tensor = self.input_tensors[virtual_pp_rank].pop()
        return input_tensor

    def _store_forward_outputs(
        self,
        virtual_pp_rank,
        output_tensor,
        schedule_chunk=None,
        loss_fn_node=None,
    ):
        self.output_tensors[virtual_pp_rank].append(output_tensor)
        # If overlap_schedule_mode eq False, the schedule chunk is a None
        self.schedule_chunks[virtual_pp_rank].append(schedule_chunk)
        if self.is_pipeline_last_stage():
            self.loss_fn_chunks.append(loss_fn_node)
            if self._forward_only:
                # no need to store tensor for backward
                if self._compute_loss:
                    self.output_tensors[virtual_pp_rank].pop()
                # save output_tensors for return value of eval batch
                else:
                    self._offload_tensors(output_tensor)
        else:
            # no need to store tensor for backward
            if self._forward_only:
                self.output_tensors[virtual_pp_rank].pop()

    def _forward_step_helper(
        self,
        micro_dataset,
        micro_step,
        overlap_schedule_mode=False,
        check_is_last_chunk=False,
    ):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=True)
        if check_is_last_chunk and virtual_pp_rank == self.num_model_chunks - 1:
            os.environ["FLAGS_last_vpp_chunk_forward"] = "1"

        self.set_virtual_pipeline_rank(virtual_pp_rank)

        input_tensor = self._get_forward_input(virtual_pp_rank)

        input_tensor_dict, use_dict = tuple_to_dict_helper(input_tensor)

        output_tensor, schedule_chunk, loss_fn_node = self._forward_step(
            input_tensor_dict if use_dict else input_tensor,
            micro_dataset,
            virtual_pp_rank,  # chunk_id
            step_id=micro_step,
            overlap_schedule_mode=overlap_schedule_mode,
        )

        output_tensor_tuple = dict_to_tuple_helper(output_tensor)

        self._store_forward_outputs(
            virtual_pp_rank, output_tensor_tuple, schedule_chunk, loss_fn_node
        )
        return output_tensor_tuple

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

    def _get_backward_input(self, virtual_pp_rank):
        # some checkers
        assert hasattr(self, 'input_tensors')
        assert hasattr(self, 'output_tensors')
        assert hasattr(self, 'output_tensor_grads')

        assert len(self.output_tensor_grads[virtual_pp_rank]) > 0, (
            f"output_tensor_grads is empty for virtual_pp_rank {virtual_pp_rank}"
        )

        assert len(self.input_tensors[virtual_pp_rank]) > 0
        assert len(self.output_tensors[virtual_pp_rank]) > 0

        input_tensor = self.input_tensors[virtual_pp_rank].pop(0)
        output_tensor = self.output_tensors[virtual_pp_rank].pop(0)
        output_tensor_grad = self.output_tensor_grads[virtual_pp_rank].pop(0)
        schedule_chunk = self.schedule_chunks[virtual_pp_rank].pop(0)
        if self.is_pipeline_last_stage():
            loss_fn_node = self.loss_fn_chunks.pop(0)
        else:
            loss_fn_node = None

        return (
            input_tensor,
            output_tensor,
            output_tensor_grad,
            schedule_chunk,
            loss_fn_node,
        )

    def _backward_step_helper(self, micro_step, overlap_schedule_mode=False):
        virtual_pp_rank = self._get_virtual_pp_rank(micro_step, forward=False)
        self.set_virtual_pipeline_rank(virtual_pp_rank)

        (
            input_tensor,
            output_tensor,
            output_tensor_grad,
            schedule_chunk,
            loss_fn_node,
        ) = self._get_backward_input(virtual_pp_rank)

        input_tensor_grad = self._backward_step(
            input_tensor,
            output_tensor,
            output_tensor_grad,
            chunk_id=virtual_pp_rank,
            step_id=micro_step,
            overlap_schedule_mode=overlap_schedule_mode,
            schedule_chunk=schedule_chunk,
            loss_fn_node=loss_fn_node,
        )

        self._overlap_comm_grads()

        return input_tensor_grad

    def _forward_backward_helper(
        self,
        micro_dataset,
        forward_micro_step_id,
        backward_micro_step_id,
        p2p_async_handle=None,
    ):
        if not self.overlap_schedule_mode:
            if p2p_async_handle is not None:
                p2p_async_handle.forward_handle_wait()

            self._record_stamp("F", forward_micro_step_id, '"B"', forward=True)
            output_tensor = self._forward_step_helper(
                micro_dataset,
                forward_micro_step_id,
            )
            self._record_stamp("F", forward_micro_step_id, '"E"', forward=True)

            if p2p_async_handle is not None:
                p2p_async_handle.forward_async_comm(output_tensor)
                p2p_async_handle.backward_handle_wait()

            # backward
            self._record_stamp(
                "B", backward_micro_step_id, '"B"', forward=False
            )
            input_tensor_grad = self._backward_step_helper(
                backward_micro_step_id,
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"E"', forward=False
            )

            if p2p_async_handle is not None:
                p2p_async_handle.backward_async_comm(input_tensor_grad)
                return
            else:
                return output_tensor, input_tensor_grad
        else:
            # 1. prepare forward inputs
            forward_virtual_pp_rank = self._get_virtual_pp_rank(
                forward_micro_step_id, forward=True
            )
            self.set_virtual_pipeline_rank(forward_virtual_pp_rank)

            if self.user_hooks_enabled:
                self.forward_hooks.run_hook()

            forward_inputs = self._get_forward_input(forward_virtual_pp_rank)

            input_tensor_dict, use_dict = tuple_to_dict_helper(forward_inputs)
            if self.is_pipeline_first_stage():
                forward_inputs = next(micro_dataset)[0]
                self._check_micro_batch_data_valid(forward_inputs)
            if self.is_pipeline_last_stage():
                labels = next(micro_dataset)[1]

            # 2. get forward chunks
            forward_chunk = self._layers.get_schedule_chunk(
                chunk_id=forward_virtual_pp_rank
            )

            if self.is_pipeline_last_stage():
                assert len(self._layers._loss_fn) == 1
                forward_loss_fn_node = self._layers._loss_fn[
                    0
                ].build_schedule_node()
                forward_loss_fn_node.labels = labels
                if self.accumulate_steps > 1 and not self._delay_scale_loss:
                    forward_loss_fn_node.scale_loss_factor = (
                        self.accumulate_steps
                    )
            else:
                forward_loss_fn_node = None

            # 3. prepare backward inputs & get backward chunks
            backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id, forward=False
            )
            self.set_virtual_pipeline_rank(backward_virtual_pp_rank)

            if self.user_hooks_enabled:
                self.backward_hooks.run_hook()

            (
                _,
                _,
                backward_grads,
                backward_chunk,
                backward_loss_fn_node,
            ) = self._get_backward_input(backward_virtual_pp_rank)

            # 4. forward & backward
            if self.processed_steps < g_profile_pipeline_details_steps:
                profile_pipeline_details(
                    "[Pipeline details] Start_forward_backward_step"
                )
            if self._enable_timer:
                self.timers("forward_backward_step").start()
            output_tensor, forward_loss, input_tensor_grad = (
                self._layers.overlapped_forward_backward(
                    forward_chunk,
                    input_tensor_dict if use_dict else forward_inputs,
                    forward_loss_fn_node,
                    backward_chunk,
                    backward_loss_fn_node,
                    backward_grads,
                    self.scaler,
                    p2p_async_handle=p2p_async_handle,
                )
            )

            output_tensor_tuple = dict_to_tuple_helper(output_tensor)

            if self.processed_steps < g_profile_pipeline_details_steps:
                profile_pipeline_details(
                    "[Pipeline details] After_forward_backward_step"
                )
            if self._enable_timer:
                self.timers("forward_backward_step").stop()

            # 5. process forward outputs
            forward_virtual_pp_rank = self._get_virtual_pp_rank(
                forward_micro_step_id, forward=True
            )
            self.set_virtual_pipeline_rank(forward_virtual_pp_rank)
            self._store_forward_outputs(
                forward_virtual_pp_rank,
                output_tensor_tuple,
                forward_chunk,
                forward_loss_fn_node,
            )

            if self.is_pipeline_first_stage() or self.is_pipeline_last_stage():
                # Only increase micro batch id at virtual first/last pp stage.
                # The micro batch id is used to load data, therefore, only increase it when load data.
                self.micro_batch_id += 1

            if self.is_pipeline_last_stage():
                # In overlap mode, only one loss_fn is supported.
                if self.total_loss is None:
                    self.total_loss = [[]]
                self.total_loss[0].append(forward_loss.detach())

            # 6. process backward outputs
            backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id, forward=False
            )
            self.set_virtual_pipeline_rank(backward_virtual_pp_rank)
            self._overlap_comm_grads()

            return output_tensor_tuple, input_tensor_grad

    def bw_hook_func(self, buffer, param):
        # For pipeline with interleave, we need to add grad to buffer without communication.
        # Use communication where appropriate to avoid dp communication and pp scheduling conflicts.
        # all reduce hook
        @paddle.autograd.no_grad()
        def fused_allreduce(*_):
            buffer.add_grad(param, use_comm=False)

        return fused_allreduce

    def register_allreduce_overlap_hook(self, model, comm_group, acc_steps, dp):
        super().register_allreduce_overlap_hook(
            model, comm_group, acc_steps, dp, group_size=sys.maxsize
        )

    def _init_buffers(self):
        # init some data buffers for interleave scheduler
        self.input_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensors = [[] for _ in range(self.num_model_chunks)]
        self.output_tensor_grads = [[] for _ in range(self.num_model_chunks)]
        self.schedule_chunks = [[] for _ in range(self.num_model_chunks)]
        self.loss_fn_chunks = []

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
        static_scheduler=False,
        return_micro_batch_loss=False,
    ):
        """
        Executes forward and backward passes for pipeline parallel training with interleaved scheduling.

        This method implements pipeline parallel training using interleaved scheduling strategy,
        inspired by Megatron-LM's implementation. It handles forward pass, backward pass, and
        gradient computation while managing communication and synchronization between stages.

        Args:
            data: Input data that will be wrapped into micro-batches
            scaler: Gradient scaler for mixed precision training
            forward_only: Whether to only perform forward pass (default: False)
            compute_loss: Whether to compute loss (default: True)
            return_micro_batch_loss: Whether to return micro-batch level loss (default: False)

        Returns:
            Training loss or logits if compute_loss is True;
            Otherwise returns output logits from the last stage

        Raises:
            AssertionError:
                - When compute_loss=False but forward_only=False
                - When cache is disabled but using interleaved pipeline
                - When buffers are not empty after execution

        Note:
            - Uses interleaved scheduling strategy (requires cache to be enabled)
            - Supports overlapping communication and computation for optimization
            - Handles startup phase, steady phase, and cooldown phase
            - Supports best unbalanced scheduler (_best_unbalanced_scheduler)
        """
        self._reset_user_hooks_status()
        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                "[Pipeline details] Start_forward_backward_step"
            )
        # use interleave scheduling strategy.
        # this strategy is inspired by:
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/schedules.py
        if not compute_loss:
            assert forward_only, (
                "compute_loss can only be set to False when forward_only is set to True"
            )

        if static_scheduler:
            assert not forward_only, (
                "static_scheduler only for training not for eval"
            )
            assert not self._profiling, (
                "While _profiling, static scheduler is not available"
            )
            if data is not None:
                warnings.warn(
                    "Static scheduler run won't real run the model, but data has been provided"
                )
            logger.info(
                "enable static_scheduler will return the pp schedule instead of the loss"
            )
            schedule = ""
        # NOTE(shenliang03): Due to ring_exchange for pipeline with interleave, cache should be enabled
        assert self._using_cache, (
            "cache should be enabled for pipeline with interleave"
        )

        self.overlap_schedule_mode = (
            hasattr(type(self._layers), "overlapped_forward_backward")
            and self._strategy.hybrid_configs[
                "pp_configs"
            ].forward_backward_overlap_scheduler
        )
        if forward_only:
            self.overlap_schedule_mode = False

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only
        self.user_hooks_enabled = not self._forward_only

        first_chunk_acc = (
            self.accumulate_steps % self.num_stages + self.num_stages
        )
        first_chunk_steps = first_chunk_acc * self.num_model_chunks
        fwd_buffer_queue = queue.Queue()
        bwd_buffer_queue = queue.Queue()
        skip_steps = self.accumulate_steps % self.num_stages
        last_stage_recv_queue = deque()

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

        def _last_stage_need_recv_next(micro_step):
            if micro_step >= first_chunk_acc:
                if len(last_stage_recv_queue) == 0:
                    return False
                else:
                    res = last_stage_recv_queue[0]
                    if micro_step - res[0] < self.num_stages:
                        return False
                    else:
                        return True
            else:
                return False

        def _last_stage_recv_pp_rank(micro_step):
            if micro_step >= first_chunk_acc:
                assert len(last_stage_recv_queue) != 0, (
                    "last_stage_recv_queue can't be empty"
                )
                virtual_pp_stage = (last_stage_recv_queue.popleft())[1]
                return virtual_pp_stage - 1
            else:
                return self.num_model_chunks - 1

        def _process_bwd_buffer(step_id, tensor):
            if self._best_unbalanced_scheduler:
                if not self.is_pipeline_first_stage():
                    bwd_buffer_queue.put(tensor)
                if step_id >= left_id and not bwd_buffer_queue.empty():
                    tensor = bwd_buffer_queue.get()
                else:
                    tensor = None
            else:
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

        self._init_buffers()

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
            startup_steps += (self.num_model_chunks - 1) * first_chunk_acc
            startup_steps = min(startup_steps, num_steps)

        # An additional micro step is needed for overplapping schedule
        if self.overlap_schedule_mode:
            startup_steps += 1
        steady_steps = num_steps - startup_steps

        for location in range(self.stage_id):
            if self.user_hooks_enabled:
                self.bubble_hooks.run_hook()

        rest_bubble_times = self.num_stages - 1 - self.stage_id

        self.set_virtual_pipeline_rank(0)
        if not static_scheduler:
            self.input_tensors[0].append(
                self._p2p_helper.recv_forward(
                    self.is_pipeline_first_stage(),
                    sync_recv=False,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
            )

        fwd_wait_handles = None
        bwd_wait_handles = None

        # run startup steps
        for micro_step in range(startup_steps):
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

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
            output_tensor = self._forward_step_helper(
                micro_dataset,
                micro_step,
                overlap_schedule_mode=self.overlap_schedule_mode,
            )
            self._record_stamp("F", micro_step, '"E"', forward=True)

            if micro_step >= startup_steps - rest_bubble_times:
                if self.user_hooks_enabled:
                    self.bubble_hooks.run_hook()

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
                # prepare for the first steady step
                if (
                    micro_step == (startup_steps - 1)
                    and (not forward_only)
                    and steady_steps
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
                        batch_p2p_comm=self._use_batch_p2p_comm,
                        skip_check_meta=not self.training,
                    )
                    # output_tensor_grad is not none if recv_next
                    # append output_tensor_grad no matter none or not
                    self.output_tensor_grads[self.num_model_chunks - 1].append(
                        output_tensor_grad
                    )
                else:
                    input_tensor = self._p2p_helper.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                        skip_check_meta=not self.training,
                    )
                # append input_tensor no matter none or not
                self.input_tensors[next_virtual_pp_rank].append(input_tensor)
            else:
                (
                    input_tensor,
                    fwd_wait_handles,
                ) = self._p2p_helper.send_forward_recv_forward(
                    output_tensor,
                    recv_prev=recv_prev,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                    overlap_p2p_comm=True,
                    skip_check_meta=not self.training,
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
                        batch_p2p_comm=self._use_batch_p2p_comm,
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

            if self._overlap_p2p_comm:
                backward_micro_step_id = micro_step

                def forward_handle_wait(fwd_wait_handles, output_tensor):
                    if fwd_wait_handles is not None:
                        for req in fwd_wait_handles:
                            req.wait()
                    self._release_output(output_tensor)

                def forward_async_comm(forward_micro_step_id, output_tensor):
                    forward_virtual_pp_rank = self._get_virtual_pp_rank(
                        forward_micro_step_id, forward=True
                    )
                    self.set_virtual_pipeline_rank(forward_virtual_pp_rank)

                    # determine whether to recv input tensor from upstream
                    recv_prev = True
                    if self.is_pipeline_first_stage(ignore_virtual=True):
                        next_forward_virtual_pp_rank = (
                            self._get_virtual_pp_rank(
                                forward_micro_step_id + 1, forward=True
                            )
                        )
                        if next_forward_virtual_pp_rank == 0:
                            # next chunk is the first chunk, not need to pre recv an input tensor
                            recv_prev = False
                    else:
                        next_forward_virtual_pp_rank = (
                            self._get_virtual_pp_rank(
                                forward_micro_step_id + 1, forward=True
                            )
                        )

                    # last iteration doesn't need recv from upstream
                    if micro_step == (steady_steps - 1):
                        recv_prev = False

                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        output_tensor = _process_fwd_buffer(
                            forward_micro_step_id, output_tensor
                        )
                    # Send activation tensor to the next stage and receive activation tensor from the
                    # previous stage
                    (
                        input_tensor,
                        fwd_wait_handles,
                    ) = self._p2p_helper.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                        overlap_p2p_comm=True,
                        skip_check_meta=not self.training,
                    )
                    return (
                        next_forward_virtual_pp_rank,
                        input_tensor,
                        fwd_wait_handles,
                    )

                def backward_handle_wait(bwd_wait_handles):
                    if bwd_wait_handles is not None:
                        for req in bwd_wait_handles:
                            req.wait()

                def backward_async_comm(
                    backward_micro_step_id, input_tensor_grad
                ):
                    if (
                        self._best_unbalanced_scheduler
                        and self.is_pipeline_last_stage(ignore_virtual=True)
                    ):
                        cur_pp_rank = self._get_virtual_pp_rank(
                            backward_micro_step_id, forward=False
                        )
                        if cur_pp_rank != 0:
                            last_stage_recv_queue.append(
                                (backward_micro_step_id, cur_pp_rank)
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
                        if self._best_unbalanced_scheduler:
                            next_backward_virtual_pp_rank = (
                                self._get_virtual_pp_rank(
                                    backward_micro_step_id + 1,
                                    forward=False,
                                )
                            )
                            if self.is_pipeline_last_stage(ignore_virtual=True):
                                recv_next = _last_stage_need_recv_next(
                                    backward_micro_step_id + 1
                                )
                        else:
                            next_backward_virtual_pp_rank = (
                                self._get_virtual_pp_rank(
                                    backward_micro_step_id + 1,
                                    forward=False,
                                )
                            )
                            if next_backward_virtual_pp_rank == (
                                self.num_model_chunks - 1
                            ):
                                # next chunk is the last chunk, not need to pre recv an output tensor grad
                                recv_next = False
                    else:
                        next_backward_virtual_pp_rank = (
                            self._get_virtual_pp_rank(
                                backward_micro_step_id + 1,
                                forward=False,
                            )
                        )

                    (
                        output_tensor_grad,
                        bwd_wait_handles,
                    ) = self._p2p_helper.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                        overlap_p2p_comm=True,
                    )
                    return (
                        next_backward_virtual_pp_rank,
                        output_tensor_grad,
                        recv_next,
                        bwd_wait_handles,
                    )

                # Package some closure functions and parameters into `P2PAsyncHandle`
                # structure to simplify function parameter passing
                p2p_async_handle = P2PAsyncHandle(
                    partial(
                        forward_handle_wait,
                        fwd_wait_handles=fwd_wait_handles,
                        output_tensor=output_tensor,
                    ),
                    partial(
                        forward_async_comm,
                        forward_micro_step_id=forward_micro_step_id,
                    ),
                    partial(
                        backward_handle_wait, bwd_wait_handles=bwd_wait_handles
                    ),
                    partial(
                        backward_async_comm,
                        backward_micro_step_id=backward_micro_step_id,
                    ),
                )

                self._forward_backward_helper(
                    micro_dataset,
                    forward_micro_step_id,
                    backward_micro_step_id,
                    p2p_async_handle,
                )

                # Information that needs to be updated
                next_forward_virtual_pp_rank = (
                    p2p_async_handle.next_forward_virtual_pp_rank
                )
                input_tensor = p2p_async_handle.input_tensor
                fwd_wait_handles = p2p_async_handle.out_fwd_wait_handles
                next_backward_virtual_pp_rank = (
                    p2p_async_handle.next_backward_virtual_pp_rank
                )
                output_tensor_grad = p2p_async_handle.output_tensor_grad
                recv_next = p2p_async_handle.recv_next
                bwd_wait_handles = p2p_async_handle.out_bwd_wait_handles
            else:
                backward_micro_step_id = micro_step
                output_tensor, input_tensor_grad = (
                    self._forward_backward_helper(
                        micro_dataset,
                        forward_micro_step_id,
                        backward_micro_step_id,
                    )
                )

                if (
                    self._best_unbalanced_scheduler
                    and self.is_pipeline_last_stage(ignore_virtual=True)
                ):
                    cur_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id, forward=False
                    )
                    if cur_pp_rank != 0:
                        last_stage_recv_queue.append(
                            (backward_micro_step_id, cur_pp_rank)
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
                if self._best_unbalanced_scheduler:
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1,
                        forward=False,
                    )
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = _last_stage_need_recv_next(
                            backward_micro_step_id + 1
                        )
                else:
                    next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                        backward_micro_step_id + 1, forward=False
                    )
                    if self.is_pipeline_last_stage(ignore_virtual=True) and (
                        next_backward_virtual_pp_rank
                        == (self.num_model_chunks - 1)
                    ):
                        # last pp stage and last virtual stage
                        recv_next = False

                (
                    input_tensor,
                    output_tensor_grad,
                ) = self._p2p_helper.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                    skip_check_meta=not self.training,
                )
            # append input_tensor no matter none or not
            self.input_tensors[next_forward_virtual_pp_rank].append(
                input_tensor
            )

            # append output_tensor_grad no matter none or not
            if self._best_unbalanced_scheduler:
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if recv_next:
                        recv_next_virtual_pp_rank = _last_stage_recv_pp_rank(
                            backward_micro_step_id + 1
                        )
                        self.output_tensor_grads[
                            recv_next_virtual_pp_rank
                        ].append(output_tensor_grad)
                        if (
                            next_backward_virtual_pp_rank
                            == self.num_model_chunks - 1
                            and recv_next_virtual_pp_rank
                            != next_backward_virtual_pp_rank
                        ):
                            self.output_tensor_grads[
                                self.num_model_chunks - 1
                            ].append(None)
                    elif (
                        next_backward_virtual_pp_rank
                        == self.num_model_chunks - 1
                    ):
                        self.output_tensor_grads[
                            self.num_model_chunks - 1
                        ].append(None)
                else:
                    self.output_tensor_grads[
                        next_backward_virtual_pp_rank
                    ].append(output_tensor_grad)
            else:
                self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                    output_tensor_grad
                )

            self._release_output(output_tensor)

        assert fwd_buffer_queue.empty(), "forward buffer should be empty"
        if not static_scheduler:
            self._release_output(output_tensor)

        # remaining backward steps
        if not forward_only:
            if self._overlap_p2p_comm and bwd_wait_handles is not None:
                for wait_handles in bwd_wait_handles:
                    wait_handles.wait()

            # no steady steps, which only occurs when accumulate_step == num_stage
            if not steady_steps:
                output_tensor_grad = self._p2p_helper.recv_backward(
                    self.is_pipeline_last_stage(),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
                self.output_tensor_grads[self.num_model_chunks - 1].append(
                    output_tensor_grad
                )
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

                if (
                    micro_step
                    < steady_steps + self.num_stages - 1 - self.stage_id
                ) and self.user_hooks_enabled:
                    self.bubble_hooks.run_hook()

                # cooldown loop
                self._record_stamp("B", micro_step, '"B"', forward=False)
                input_tensor_grad = self._backward_step_helper(
                    micro_step, overlap_schedule_mode=self.overlap_schedule_mode
                )
                self._record_stamp("B", micro_step, '"E"', forward=False)
                next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                    micro_step + 1,
                    forward=False,
                )
                if (
                    self._best_unbalanced_scheduler
                    and self.is_pipeline_last_stage(ignore_virtual=True)
                ):
                    cur_pp_rank = self._get_virtual_pp_rank(
                        micro_step, forward=False
                    )
                    if cur_pp_rank != 0:
                        last_stage_recv_queue.append((micro_step, cur_pp_rank))

                recv_next = True
                if self.is_pipeline_last_stage(ignore_virtual=True):
                    if self._best_unbalanced_scheduler:
                        recv_next = _last_stage_need_recv_next(micro_step + 1)
                    else:
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

                # append output_tensor_grad no matter none or not
                if self._best_unbalanced_scheduler:
                    if self.is_pipeline_last_stage(ignore_virtual=True):
                        output_tensor_grad = (
                            self._p2p_helper.send_backward_recv_backward(
                                input_tensor_grad,
                                recv_next=recv_next,
                                batch_p2p_comm=self._use_batch_p2p_comm,
                            )
                        )
                        if recv_next:
                            recv_next_virtual_pp_rank = (
                                _last_stage_recv_pp_rank(micro_step + 1)
                            )
                            self.output_tensor_grads[
                                recv_next_virtual_pp_rank
                            ].append(output_tensor_grad)
                        else:
                            self.output_tensor_grads[
                                next_backward_virtual_pp_rank
                            ].append(output_tensor_grad)
                    else:
                        self.output_tensor_grads[
                            next_backward_virtual_pp_rank
                        ].append(
                            self._p2p_helper.send_backward_recv_backward(
                                input_tensor_grad,
                                recv_next=recv_next,
                                batch_p2p_comm=self._use_batch_p2p_comm,
                            )
                        )
                else:
                    self.output_tensor_grads[
                        next_backward_virtual_pp_rank
                    ].append(
                        self._p2p_helper.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            batch_p2p_comm=self._use_batch_p2p_comm,
                        )
                    )

            self._sync_overlap_grads()

            for _ in range(self.stage_id):
                self.bubble_hooks.run_hook()

            if static_scheduler:
                self._reset_counter()
                return schedule

            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").start()
            self._layers.allreduce_shared_weight_gradients()
            if self._enable_timer:
                self.timers("allreduce_shared_weight_gradients").stop()

        self._flush_records()

        assert bwd_buffer_queue.empty(), "backward buffer should be empty"
        if compute_loss:
            # return loss if compute loss
            if self._enable_timer:
                self.timers("broadcast_final_loss").start()
            with paddle.amp.auto_cast(enable=False):
                train_loss_or_logits = self._broadcast_final_loss(
                    return_micro_batch_loss
                )
            if self._enable_timer:
                self.timers("broadcast_final_loss").stop()
        else:
            # else just return logits without loss func calc
            train_loss_or_logits = self.output_tensors.pop()

        if self._clear_every_step_cache:
            self._p2p_helper.clear_meta_cache()

        self.timer_printer()
        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                "[Pipeline details] End_forward_backward_step"
            )
        self.processed_steps += 1
        self._check_user_hooks_status_at_step_end()

        # reset dynamic meta counter
        if self._dynamic_shape:
            assert self._p2p_helper._dynamic_cnt == len(
                self._p2p_helper._send_recv_meta_list
            ), "p2p dynamic_cnt should equal to send_recv_meta_list"
            self._p2p_helper._dynamic_cnt = 0

        return train_loss_or_logits

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
        loss_fn_idx=0,
        return_micro_batch_loss=False,
    ):
        """
        Execute one training batch with pipeline parallel interleaving schedule.

        Performs forward/backward passes and optimizer update for a batch of data
        using pipeline parallel with interleaved scheduling.

        Args:
            data: Input data for the batch
            optimizer: Optimizer instance for parameter updates
            lr_scheduler: Learning rate scheduler (optional)
            scaler: Gradient scaler for mixed precision training (optional)
            loss_fn_idx: Index of loss function to use (default: 0)
            return_micro_batch_loss: Whether to return per-micro-batch losses (default: False)

        Returns:
            The computed training loss. If return_micro_batch_loss is True,
            returns a tuple of (total_loss, micro_batch_losses).

        Note:
            - Handles both FP16/FP32 mixed precision training when scaler is provided
            - Supports multiple loss functions through loss_fn_idx
            - Uses interleaved pipeline parallel schedule for efficient training
        """
        data = self._prepare_training(data, optimizer, lr_scheduler)

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        # interleave scheduler for pipeline parallel
        train_loss = self.forward_backward_pipeline(
            data, scaler, return_micro_batch_loss=return_micro_batch_loss
        )

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

    def eval_batch(
        self, data, compute_loss=False, loss_fn_idx=0, return_host_tensor=False
    ):
        self.user_hooks_enabled = False
        # reset the virtual pp rank for each run
        self.set_virtual_pipeline_rank(0)

        self._layers.eval()
        origin_compute_loss = self._compute_loss
        self._compute_loss = compute_loss
        origin_return_host_tensor = self._return_host_tensor
        self._return_host_tensor = return_host_tensor

        # check loss_fn_idx is valid and loss_fn exists
        assert (
            loss_fn_idx in range(len(self._layers._loss_fn))
            and self._layers._loss_fn[loss_fn_idx] is not None
        ), f"loss function {loss_fn_idx} should exist to compute loss"
        self.loss_fn_idx = loss_fn_idx

        train_loss_or_logits = self.forward_backward_pipeline(
            data, None, forward_only=True, compute_loss=compute_loss
        )
        self._init_buffers()
        self._compute_loss = origin_compute_loss
        self._return_host_tensor = origin_return_host_tensor
        return train_loss_or_logits

    def get_static_scheduler(self):
        return self.forward_backward_pipeline(
            data=None, scaler=None, static_scheduler=True
        )
