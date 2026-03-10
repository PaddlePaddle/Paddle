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
"""Pipeline Parallel with Balanced Memory Implementation.

This module contains the VPPFhenBInBalancedMemory class which implements
pipeline parallelism with memory-optimized schedule.
"""

from __future__ import annotations

import paddle

from .fthenb_pipeline import PipelineParallelWithInterleaveFthenB
from .p2p_handle import OffloadQueue
from .utils import profile_pipeline_details

g_profile_pipeline_details_steps = int(
    paddle.base.core.os.getenv("FLAGS_profile_pipeline_details_steps", "0")
)


class VPPFhenBInBalancedMemory(PipelineParallelWithInterleaveFthenB):
    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        self.overlap_schedule_mode = False

    def _get_scheduler_name(self):
        return "VPPFhenBInBalancedMemory"

    def _init_user_bubble_hooks(self):
        self.bubble_hooks = PipelineHook()
        self.bubble_hooks.set_hooks_capacity(2 * self.num_stages - 2)

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
        return_micro_batch_loss=False,
    ):
        self._reset_user_hooks_status()
        if not compute_loss:
            assert forward_only, (
                "compute_loss can only be set to False when forward_only is set to True"
            )
        assert self._using_cache, (
            "cache should be enabled for pipeline with interleave"
        )
        self.user_hooks_enabled = not forward_only
        if forward_only:
            return super().forward_backward_pipeline(
                data,
                scaler,
                forward_only,
                compute_loss,
                return_micro_batch_loss,
            )

        if self.processed_steps < g_profile_pipeline_details_steps:
            profile_pipeline_details(
                "[Pipeline details] Start_forward_backward_step"
            )

        # init some attributes for this batch run
        self.scaler = scaler
        self.total_loss = None
        self.micro_batch_id = 0
        self._forward_only = forward_only

        self._init_buffers()

        backward_send_recv_buffer_queue = OffloadQueue(
            offload=self._enable_offload_queue
        )
        forward_send_recv_buffer_queue = OffloadQueue(
            offload=self._enable_offload_queue
        )

        skip_steps = self.accumulate_steps - self.num_stages
        micro_dataset = self._wrap_data(data)
        num_steps = self.accumulate_steps * self.num_model_chunks

        # the whole pipeline is splited into 3 parse:
        # startup_steps, steady_1f1b_steps, cooldown_steps
        startup_steps = (
            self.accumulate_steps * (self.num_model_chunks - 1)
            + self.num_stages
            - self.stage_id
            - 1
        )
        steady_1f1b_steps = self.accumulate_steps - (
            self.num_stages - self.stage_id - 1
        )
        cooldown_steps = startup_steps

        # Bubbles before startup_steps
        for _ in range(self.stage_id):
            if self.user_hooks_enabled:
                self.bubble_hooks.run_hook()

        self.set_virtual_pipeline_rank(0)
        self.input_tensors[0].append(
            self._p2p_helper.recv_forward(
                self.is_pipeline_first_stage(),
                sync_recv=False,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
        )

        # In startup_steps, we send every output_tensor of last stage,
        # to simplify the code logic of stage 1F1B.
        for micro_step in range(startup_steps):
            self._record_stamp("F", micro_step, '"B"', forward=True)
            output_tensor = self._forward_step_helper(micro_dataset, micro_step)
            self._record_stamp("F", micro_step, '"E"', forward=True)
            next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                micro_step + 1, forward=True
            )
            recv_prev = True
            if self.is_pipeline_first_stage(ignore_virtual=True) and (
                micro_step < self.num_stages - 1
            ):
                recv_prev = False

            input_tensor = self._p2p_helper.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                batch_p2p_comm=self._use_batch_p2p_comm,
                skip_check_meta=not self.training,
            )
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if input_tensor is not None:
                    # stash the input_tensor and it will be used in the next chunk later
                    forward_send_recv_buffer_queue.put(input_tensor)
                if next_forward_virtual_pp_rank == 0:
                    input_tensor = None
                else:
                    # when a input_tensor is needed, get one from the queue
                    input_tensor = forward_send_recv_buffer_queue.get()

            self.input_tensors[next_forward_virtual_pp_rank].append(
                input_tensor
            )
            self._release_output(output_tensor)

        if self.is_pipeline_first_stage(ignore_virtual=True):
            assert (
                forward_send_recv_buffer_queue.qsize()
                == num_steps - startup_steps - 1
            ), forward_send_recv_buffer_queue.qsize()

        input_tensor_grad = None
        for micro_step in range(steady_1f1b_steps):
            first_iter = micro_step == 0
            last_iter = micro_step == (steady_1f1b_steps - 1)
            forward_micro_step_id = micro_step + startup_steps
            backward_micro_step_id = micro_step

            self._record_stamp("F", forward_micro_step_id, '"B"', forward=True)
            output_tensor = self._forward_step_helper(
                micro_dataset,
                forward_micro_step_id,
                check_is_last_chunk=True,
            )
            self._record_stamp("F", forward_micro_step_id, '"E"', forward=True)

            if first_iter:
                for _ in range(self.num_stages - self.stage_id - 1):
                    if self.user_hooks_enabled:
                        self.bubble_hooks.run_hook()

            # NOTE: `send_forward_recv_backward` is intentionally unused to
            # prevent hanging bugs in dynamic shape mode.
            self._p2p_helper.send_forward(
                output_tensor,
                self.is_pipeline_last_stage(ignore_virtual=True),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
            output_tensor_grad = self._p2p_helper.recv_backward(
                self.is_pipeline_last_stage(ignore_virtual=True),
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
            # Unlike normal FthenB, in 1F1B steps, we recv output_tensor_grad
            # for the current step, but not for the next step
            cur_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id, forward=False
            )
            self.output_tensor_grads[cur_backward_virtual_pp_rank].append(
                output_tensor_grad
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"B"', forward=False
            )
            input_tensor_grad = self._backward_step_helper(
                backward_micro_step_id
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"E"', forward=False
            )

            # stash the input_tensor_grad and it will be sent to ths last stage later
            if self.is_pipeline_first_stage(ignore_virtual=True):
                backward_send_recv_buffer_queue.put(input_tensor_grad)

            if not last_iter:
                # NOTE: `send_backward_recv_forward` is intentionally unused to
                # prevent hanging bugs in dynamic shape mode.
                input_tensor = self._p2p_helper.recv_forward(
                    self.is_pipeline_first_stage(ignore_virtual=True),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
                self._p2p_helper.send_backward(
                    input_tensor_grad,
                    self.is_pipeline_first_stage(ignore_virtual=True),
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
                next_forward_virtual_pp_rank = self._get_virtual_pp_rank(
                    forward_micro_step_id + 1, forward=True
                )
                if self.is_pipeline_first_stage(ignore_virtual=True):
                    input_tensor = forward_send_recv_buffer_queue.get()
                self.input_tensors[next_forward_virtual_pp_rank].append(
                    input_tensor
                )
            else:
                for _ in range(self.num_stages - self.stage_id - 1):
                    if self.user_hooks_enabled:
                        self.bubble_hooks.run_hook()

        assert forward_send_recv_buffer_queue.qsize() == 0, (
            forward_send_recv_buffer_queue.qsize()
        )

        next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
            steady_1f1b_steps, forward=False
        )

        # no more fwd, but we need to send the input_tensor_grad.
        if self.is_pipeline_first_stage(ignore_virtual=True):
            input_tensor_grad = backward_send_recv_buffer_queue.get()
        self.output_tensor_grads[next_backward_virtual_pp_rank].append(
            self._p2p_helper.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=True,
                batch_p2p_comm=self._use_batch_p2p_comm,
            )
        )

        # run cooldown
        for micro_step in range(cooldown_steps):
            backward_micro_step_id = micro_step + steady_1f1b_steps

            self._record_stamp(
                "B", backward_micro_step_id, '"B"', forward=False
            )
            input_tensor_grad = self._backward_step_helper(
                backward_micro_step_id
            )
            self._record_stamp(
                "B", backward_micro_step_id, '"E"', forward=False
            )
            next_backward_virtual_pp_rank = self._get_virtual_pp_rank(
                backward_micro_step_id + 1, forward=False
            )

            recv_next = True
            if backward_micro_step_id == (num_steps - 1):
                recv_next = False
            if self.is_pipeline_first_stage(ignore_virtual=True):
                if not self.is_pipeline_first_stage():
                    backward_send_recv_buffer_queue.put(input_tensor_grad)

                if (
                    self.is_pipeline_first_stage()
                    and backward_micro_step_id % self.accumulate_steps
                    >= skip_steps
                ):
                    # no need to send the input_tensor_grad anymore
                    input_tensor_grad = None
                else:
                    input_tensor_grad = backward_send_recv_buffer_queue.get()
            self.output_tensor_grads[next_backward_virtual_pp_rank].append(
                self._p2p_helper.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    batch_p2p_comm=self._use_batch_p2p_comm,
                )
            )

        assert backward_send_recv_buffer_queue.empty(), (
            "send_recv buffer should be empty"
        )

        # Bubbles after cooldown
        for _ in range(self.stage_id):
            if self.user_hooks_enabled:
                self.bubble_hooks.run_hook()

        # reset dynamic meta counter
        if self._dynamic_shape:
            assert self._p2p_helper._dynamic_cnt == len(
                self._p2p_helper._send_recv_meta_list
            ), "p2p dynamic_cnt should equal to send_recv_meta_list"
            self._p2p_helper._dynamic_cnt = 0

        self._flush_records()
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
