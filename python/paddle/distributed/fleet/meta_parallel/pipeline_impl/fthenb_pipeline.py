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
"""Pipeline Parallel with Interleave FthenB Implementation.

This module contains the PipelineParallelWithInterleaveFthenB class which implements
pipeline parallelism with forward-then-backward schedule in interleave mode.
"""

from __future__ import annotations

import paddle

from .interleave_pipeline import PipelineParallelWithInterleave
from .utils import profile_pipeline_details

g_profile_pipeline_details_steps = int(
    paddle.base.core.os.getenv("FLAGS_profile_pipeline_details_steps", "0")
)


class PipelineParallelWithInterleaveFthenB(PipelineParallelWithInterleave):
    def __init__(self, layers, hcg, strategy):
        # Initialize the basic parameters of the parent class PipelineParallel
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        # Whether to enable overlapped scheduling mode (disabled by default)
        self.overlap_schedule_mode = False

    def _get_scheduler_name(self):
        return "PipelineParallelWithInterleaveFthenB"

    def _init_user_bubble_hooks(self):
        # (TODO:gexiao) support bubble hooks if needed
        self.bubble_hooks = None
        # self.bubble_hooks = PipelineHook()
        # self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def _check_sanity(self):
        assert framework.in_dynamic_mode(), (
            "virtual pipeline stage with interleave only support eager dygraph mode"
        )

        assert self.num_stages > 2, (
            "virtual pipeline must run under pp degree > 2"
        )

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
            for buffer in self._chunk_2_comm_buffers[chunk_idx]:
                buffer.comm_grads()

        if self.stage_id == 0:
            return

        if (
            self._backward_step_count
            == self.accumulate_steps * self._virtual_pp_world_size
        ):
            for buffer in self._chunk_2_comm_buffers[0]:
                buffer.comm_grads()

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
                buffer.scale_grads()

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
        return_micro_batch_loss=False,
    ):
        self._reset_user_hooks_status()
        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                "[Pipeline details] Start_forward_backward_step"
            )
        if not compute_loss:
            assert forward_only, (
                "compute_loss can only be set to False when forward_only is set to True"
            )

        # NOTE(shenliang03): Due to ring_exchange for pipeline with interleave, cache should be enabled
        assert self._using_cache, (
            "cache should be enabled for pipeline with interleave"
        )

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only
        self.user_hooks_enabled = not self._forward_only

        assert (
            self.accumulate_steps == self.num_stages
            or self.accumulate_steps % self.num_stages == 0
        ), (
            f"accumulate_steps({self.accumulate_steps}) and num_stages({self.num_stages}) should be a multiple or accumulate_steps % num_stages == 0"
        )

        self._backward_step_count = 0
        skip_steps = self.accumulate_steps - self.num_stages
        send_recv_buffer_queue = queue.Queue()

        self._init_buffers()

        micro_dataset = self._wrap_data(data)
        num_steps = self.accumulate_steps * self.num_model_chunks

        self.set_virtual_pipeline_rank(0)
        self.input_tensors[0].append(
            self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                sync_recv=False,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
        )

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
                output_tensor,
                recv_prev=recv_prev,
                batch_p2p_comm=self._use_batch_p2p_comm,
                skip_check_meta=not self.training,
            )
            self.input_tensors[next_virtual_pp_rank].append(input_tensor)

            self._release_output(output_tensor)

        assert send_recv_buffer_queue.empty(), (
            "send_recv buffer should be empty"
        )

        # remaining backward steps
        if not forward_only:
            self.output_tensor_grads[self.num_model_chunks - 1].append(
                self._p2p_helper.recv_backward(
                    self.is_pipeline_last_stage(),
                    sync_recv=False,
                    batch_p2p_comm=self._use_batch_p2p_comm,
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
                        input_tensor_grad,
                        recv_next=recv_next,
                        batch_p2p_comm=self._use_batch_p2p_comm,
                    )
                )

            assert send_recv_buffer_queue.empty(), (
                "send_recv buffer should be empty"
            )

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
        return train_loss_or_logits
