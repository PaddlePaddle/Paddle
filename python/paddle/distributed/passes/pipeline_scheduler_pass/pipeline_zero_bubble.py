# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from collections import deque

import paddle
from paddle.base import core

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import (
    PipelineMemoryEstimator,
    _program_for_zero_bubble,
    _program_for_zero_bubble_vpp,
    split_matmul_grad_to_matmul,
)
from .pipeline_pass_base import PipelinePassBase

FORWARD = "forward"
BACKWARD = "backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


class PipelineZeroBubbleBase(PipelinePassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("enable_optimizer_post_validation", 0)

    def _split_matmul_grad_ops_to_matmul(self, program, dist_context):
        for block in program.blocks:
            matmul_grad_op_idx = []
            ops = block.ops
            for i, op_i in enumerate(ops):
                if (
                    op_i.type == "matmul_v2_grad"
                    and not op_i.attr("trans_x")
                    and not op_i.attr("trans_y")
                ):
                    matmul_grad_op_idx.append(i)

            for matmul_grad_id in reversed(matmul_grad_op_idx):
                split_matmul_grad_to_matmul(
                    block, matmul_grad_id, dist_context=dist_context
                )


@register_pass("pipeline_scheduler_ZBH1")
class PipelineZeroBubblePipelinePass(PipelineZeroBubbleBase):
    def __init__(self):
        super().__init__()

    def _create_job_list(self):
        num_micro_batches = self.get_attr("num_micro_batches")
        pp_stage = self.get_attr("pp_stage")
        pp_degree = self.get_attr("pp_degree")

        job_list = []
        assert (
            pp_degree <= num_micro_batches
        ), "Num of micro batches should larger than or equal to pp degree."

        micro_batch_in_warmup = pp_degree - pp_stage
        micro_batch_in_zero_bubble = num_micro_batches - pp_degree

        forward_micro_batch_id = 0
        for _ in range(micro_batch_in_warmup):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        backward_micro_batch_id = 0
        for _ in range(pp_stage):
            backward_b_job = core.Job(BACKWARD + '_b')
            backward_b_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_b_job)
            backward_micro_batch_id += 1

            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        for _ in range(micro_batch_in_zero_bubble):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)

            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)

            forward_micro_batch_id += 1
            backward_micro_batch_id += 1

        for _ in range(micro_batch_in_warmup - 1):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1

        if pp_stage > 0:
            backward_b_job = core.Job(BACKWARD + '_b')
            backward_b_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_b_job)

            backward_w_job = core.Job(BACKWARD + '_w')
            backward_w_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_w_job)
        else:
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
        backward_micro_batch_id += 1

        for i in range(pp_stage):
            backward_w_job = core.Job(BACKWARD + '_w')
            backward_w_job.set_micro_batch_id(i)
            job_list.append(backward_w_job)

        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        self._split_matmul_grad_ops_to_matmul(program, dist_context)
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        types, sub_program_list = _program_for_zero_bubble(
            program, enable_send_recv_overlap
        )
        return types, sub_program_list


@register_pass("pipeline_scheduler_ZBVPP")
class PipelineZeroBubbleVirtualPipelinePass(PipelineZeroBubblePipelinePass):
    def __init__(self):
        super().__init__()
        self.set_attr("enable_optimizer_post_validation", 0)
        self.program_mem_usages = []
        self.program_max_mem_usages = []
        self.base_memory = []
        self.program_runtime = {
            "forward": 1000,
            "backward_b": 1000,
            "backward_w": 1000,
            "communication": 1,
        }

    def _create_job_list(self):
        v_scheduler = VScheduleCreator(
            self.get_attr("pp_degree"),
            self.get_attr("num_micro_batches"),
            self.get_attr("vpp_degree"),
            self.program_mem_usages,
            self.program_max_mem_usages,
            self.base_memory,
            self.program_runtime,
        )

        schedule, max_bubble = None, None
        for fill_w_before_b in [True, False]:
            for fill_w_before_f in [True, False]:
                if schedule is None:
                    schedule, _, max_bubble = v_scheduler.create_v_schedule(
                        fill_w_before_b=fill_w_before_b,
                        fill_w_before_f=fill_w_before_f,
                    )
                else:
                    (
                        new_schedule,
                        _,
                        new_max_bubble,
                    ) = v_scheduler.create_v_schedule(
                        fill_w_before_b=fill_w_before_b,
                        fill_w_before_f=fill_w_before_f,
                    )
                    if new_max_bubble < max_bubble:
                        schedule, max_bubble = new_schedule, new_max_bubble

        stage_schedule = schedule[self.get_attr("pp_stage")]
        job_list = []

        for job_info in stage_schedule:
            job = core.Job(job_info["type"])
            job.set_micro_batch_id(job_info["micro_batch"])
            job_list.append(job)

        return job_list

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        num_model_chunks = self.get_attr("vpp_degree")

        self._split_matmul_grad_ops_to_matmul(program, dist_context)
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        types, sub_program_list = _program_for_zero_bubble_vpp(
            program, num_model_chunks, dist_context, enable_send_recv_overlap
        )
        self._estimate_program_mem_usagess(
            types, sub_program_list, dist_context
        )
        self._get_all_device_base_memory()

        return types, sub_program_list

    def _estimate_program_mem_usagess(
        self, types, sub_program_list, dist_context
    ):
        types = types[:-1]
        type_to_program = dict(zip(types, sub_program_list))
        memory_estimator = PipelineMemoryEstimator()
        memory_estimator.set_program_skip_gc_vars(type_to_program, types)

        mem_usages = []
        max_mem_usages = []
        for type in types:
            mem_usage, max_mem_usage = memory_estimator.estimate_memory(
                type_to_program[type], type, dist_context
            )
            mem_usages.append(mem_usage)
            max_mem_usages.append(max_mem_usage)

        # Get program memory usage from all devices
        paddle.disable_static()
        all_mem_usages = []
        all_max_usages = []
        paddle.distributed.all_gather_object(all_mem_usages, mem_usages)
        paddle.distributed.all_gather_object(all_max_usages, max_mem_usages)
        paddle.enable_static()

        self.program_mem_usages = [{} for _ in range(len(all_mem_usages))]
        self.program_max_mem_usages = [{} for _ in range(len(all_max_usages))]

        for id in range(len(all_mem_usages)):
            for i, type in enumerate(types):
                self.program_mem_usages[id][type] = all_mem_usages[id][i]
                self.program_max_mem_usages[id][type] = all_max_usages[id][i]

    def _get_all_device_base_memory(self):
        paddle.disable_static()
        self.base_memory = []
        rank = self.get_attr("pp_stage")
        base_memory = paddle.device.cuda.memory_allocated(rank)
        paddle.distributed.all_gather_object(self.base_memory, base_memory)
        paddle.enable_static()


class VScheduleCreator:
    def __init__(
        self,
        num_stage,
        num_micro_batch,
        num_model_chunks,
        program_mem_usages,
        program_max_mem_usages,
        base_memory,
        program_runtime,
        max_memory=None,
    ):
        self.num_stage = num_stage
        self.num_micro_batch = num_micro_batch
        self.num_model_chunks = num_model_chunks
        self.num_nodes = num_model_chunks * num_stage * num_micro_batch * 3
        self.program_mem_usages = program_mem_usages
        self.program_max_mem_usages = program_max_mem_usages
        self.job_types = ["forward", "backward_w", "backward_b"]
        self.program_runtime = program_runtime
        self.base_memory = base_memory
        self.max_memory = max_memory
        if max_memory is None:
            self.max_memory = float("inf")

    def init_schedule(self):
        job_counter = {}
        for job_type in self.job_types:
            for chunk_id in range(self.num_model_chunks):
                job_counter[f"{job_type}{chunk_id}"] = 0

        self._job_counters = [job_counter.copy() for _ in range(self.num_stage)]
        self._job_end_times = [-1] * self.num_nodes
        self._stage_current_time = [0] * self.num_stage
        self._stage_mem_usage = self.base_memory.copy()
        self._pending_w = [deque() for _ in range(self.num_stage)]
        self._stage_job_schedule = [[] for _ in range(self.num_stage)]
        self._stage_bubbles = [0] * self.num_stage

    def create_v_schedule(
        self, fill_w_before_f=True, fill_w_before_b=True, approved_bubbles=None
    ):
        self.init_schedule()
        if approved_bubbles is None:
            approved_bubbles = [-1] * self.num_stage
        max_approved_bubble = max(approved_bubbles)

        self._insert_forward_jobs_before_forward1()
        self._insert_forward_jobs_before_backward_b()
        self._insert_jobs_after_backward_start(
            fill_w_before_f, fill_w_before_b, approved_bubbles
        )

        schedule = self._stage_job_schedule.copy()
        end_time = self._job_end_times.copy()
        max_bubble = self._get_max_stage_bubble()

        if max_approved_bubble < 0 or max_bubble < max_approved_bubble:
            new_schedule, new_end_time, new_max_bubble = self.create_v_schedule(
                fill_w_before_f, fill_w_before_b, self._stage_bubbles
            )

            if new_max_bubble < max_bubble:
                return new_schedule, new_end_time, new_max_bubble

        return schedule, end_time, max_bubble

    def _insert_forward_jobs_before_forward1(self):
        # Step1: Insert forward jobs with chunk_id=0 into the schedule
        for i in range(self.num_stage):
            self._put_job_into_schedule("forward", chunk_id=0, stage_id=i)

        # Step2: Insert forward jobs with chunk_id=1 into the schedule
        for i in range(self.num_stage - 1, -1, -1):
            if i == self.num_stage - 1:
                self._put_job_into_schedule("forward", chunk_id=1, stage_id=i)
                continue

            forward1_start_time = (
                self._job_end_times[self._get_job_id("forward", 1, i + 1, 0)]
                + self.program_runtime["communication"]
            )

            self._fill_bubble_with_forward(
                [0], i, forward1_start_time, insert_order="down"
            )
            self._put_job_into_schedule("forward", chunk_id=1, stage_id=i)

    def _insert_forward_jobs_before_backward_b(self):
        for chunk_id in range(2, self.num_model_chunks):
            stage_order = list(range(self.num_stage))
            forward_insert_order = "up"
            if chunk_id % 2:
                stage_order = stage_order[::-1]
                forward_insert_order = "down"

            self._put_job_into_schedule("forward", chunk_id, stage_order[0])
            for i in stage_order[1:]:
                next_chunk_forward_start_time = (
                    self._job_end_times[
                        self._get_job_id(
                            "forward", chunk_id, stage_order[i - 1], 0
                        )
                    ]
                    + self.program_runtime["communication"]
                )

                fill_chunk_ids = list(range(0, chunk_id))
                self._fill_bubble_with_forward(
                    fill_chunk_ids,
                    i,
                    next_chunk_forward_start_time,
                    insert_order=forward_insert_order,
                )

                self._put_job_into_schedule("forward", chunk_id, i)

        stage_order = list(range(self.num_stage))
        forward_insert_order = "up"

        if self.num_model_chunks % 2:
            stage_order = stage_order[::-1]
            forward_insert_order = "down"

        backward_b_start_time = self._job_end_times[
            self._get_job_id(
                "forward", self.num_model_chunks - 1, stage_order[0], 0
            )
        ]

        for i in stage_order[1:]:
            backward_b_start_time = (
                backward_b_start_time
                + self.program_runtime["communication"]
                + self.program_runtime["backward_b"]
            )

            fill_chunk_ids = list(range(0, self.num_model_chunks))
            self._fill_bubble_with_forward(
                fill_chunk_ids,
                i,
                backward_b_start_time,
                insert_order=forward_insert_order,
            )

        # # Step1: Insert forward jobs after forward1 to fill the bubble
        # chunk_id = 0
        # stage_order = list(range(1, self.num_stage))
        # for i, stage_id in enumerate(stage_order):
        #     prev_stage_id = self._get_prev_stage_id("forward", chunk_id, stage_id)

        #     # Each stage's forward job number should not be less than the previous stage
        #     while (
        #         self._get_stage_micro_batch_id(stage_id, "forward")
        #         < self._get_stage_micro_batch_id(prev_stage_id, "forward")
        #     ) or self._compare_forward_micro_batch_id(stage_id, prev_stage_id, 1):
        #         insert_order = stage_order[i:][::-1]

        #         for id in insert_order:
        #             if self._micro_batch_id_check("forward", chunk_id, id):
        #                 self._put_job_into_schedule("forward", chunk_id, id)
        #         chunk_id = (chunk_id + 1) % 2

        # # Step2: Insert forward jobs before backward_b
        # for stage_id in range(self.num_stage - 1, -1, -1):

    def _insert_jobs_after_backward_start(
        self, fill_w_before_f, fill_w_before_b, approved_bubbles
    ):
        backward_b_job_number = self.num_model_chunks * self.num_micro_batch

        for _ in range(backward_b_job_number):
            # Step1: Check memory usage, if not enough, put pending backward_w job into schedule
            for stage_id in range(self.num_stage):
                while not self._memory_check("backward_b", 0, stage_id):
                    if len(self._pending_w[stage_id]) == 0:
                        raise ValueError(
                            "No pending backward_w job and backward_b job exceeds the memory limit."
                        )
                    self._put_w_job_into_schedule(stage_id)

            # Step2: Insert backward_b job for each stage
            b_ranks = [[] for _ in range(self.num_model_chunks)]
            for stage_id in range(self.num_stage):
                for chunk_id in range(0, self.num_model_chunks):
                    if self._can_schedule_b_task(stage_id, chunk_id):
                        b_ranks[chunk_id].append(stage_id)
                        break

            for chunk_id, b_rank in enumerate(b_ranks):
                if chunk_id % 2 == 0:
                    b_rank = b_rank[::-1]

                for stage_id in b_rank:
                    dependency_job_end_time = self._get_dependency_job_end_time(
                        "backward_b",
                        chunk_id,
                        stage_id,
                        self._job_counters[stage_id][f"backward_b{chunk_id}"],
                    )
                    while (
                        len(self._pending_w[stage_id])
                        and dependency_job_end_time
                        + self.program_runtime["communication"]
                        >= self._stage_current_time[stage_id]
                        + self.program_runtime["backward_w"]
                    ):
                        self._put_w_job_into_schedule(stage_id)

                    max_stage_bubble = self._get_max_stage_bubble(
                        stage_id, approved_bubbles
                    )
                    stage_bubble = self._stage_bubbles[stage_id]
                    if (
                        len(self._pending_w[stage_id])
                        and dependency_job_end_time
                        + self.program_runtime["communication"]
                        - self._stage_current_time[stage_id]
                        > max_stage_bubble - stage_bubble
                    ):
                        if chunk_id == 0 or fill_w_before_b:
                            self._put_w_job_into_schedule(stage_id)

                    self._put_job_into_schedule(
                        "backward_b", chunk_id, stage_id
                    )

            # Step3: Insert forward jobs after backward_b
            for stage_id in range(self.num_stage):
                for chunk_id in range(self.num_model_chunks - 1, -1, -1):
                    if self._can_schedule_f_task(stage_id, chunk_id):
                        while (
                            self._stage_mem_usage[stage_id]
                            + self.program_max_mem_usages[stage_id][
                                f"forward{chunk_id}"
                            ]
                            > self.max_memory
                        ):
                            if len(self._pending_w[stage_id]) == 0:
                                raise ValueError(
                                    "No pending backward_w job and forward job exceeds the memory limit."
                                )
                            self._put_w_job_into_schedule(stage_id)

                        dependency_job_end_time = (
                            self._get_dependency_job_end_time(
                                "forward",
                                chunk_id,
                                stage_id,
                                self._job_counters[stage_id][
                                    f"forward{chunk_id}"
                                ],
                            )
                        )
                        while (
                            len(self._pending_w[stage_id])
                            and dependency_job_end_time
                            + self.program_runtime["communication"]
                            >= self._stage_current_time[stage_id]
                            + self.program_runtime["backward_w"]
                        ):
                            self._put_w_job_into_schedule(stage_id)

                        max_stage_bubble = self._get_max_stage_bubble(
                            stage_id, approved_bubbles
                        )
                        stage_bubble = self._stage_bubbles[stage_id]
                        if (
                            len(self._pending_w[stage_id])
                            and dependency_job_end_time
                            + self.program_runtime["communication"]
                            - self._stage_current_time[stage_id]
                            > max_stage_bubble - stage_bubble
                        ):
                            if fill_w_before_f:
                                self._put_w_job_into_schedule(stage_id)

                        self._put_job_into_schedule(
                            "forward", chunk_id, stage_id
                        )
                        break

        for stage_id in range(self.num_stage):
            while len(self._pending_w[stage_id]):
                self._put_w_job_into_schedule(stage_id)

    def _can_schedule_f_task(self, stage_id, chunk_id):
        return self._can_schedule_task("forward", chunk_id, stage_id)

    def _can_schedule_b_task(self, stage_id, chunk_id):
        if chunk_id == self.num_model_chunks - 1:
            if (
                self._job_counters[stage_id][f"backward_b{chunk_id}"]
                < self.num_micro_batch
            ):
                return True
            return False

        if (
            self._job_counters[stage_id][f"backward_b{chunk_id + 1}"]
            == self.num_micro_batch
        ):
            return True

        return self._can_schedule_task("backward_b", chunk_id, stage_id)

    def _can_schedule_task(self, job_type, chunk_id, stage_id):
        if job_type == "forward":
            current_key = f"forward{chunk_id}"
            prev_key = f"forward{chunk_id - 1}"
        elif job_type == "backward_b":
            current_key = f"backward_b{chunk_id}"
            prev_key = f"backward_b{chunk_id + 1}"

        micro_batch_id = self._job_counters[stage_id][current_key]
        if micro_batch_id >= self.num_micro_batch:
            return False

        if (job_type == "forward" and chunk_id > 0) or (
            job_type == "backward_b" and chunk_id < self.num_model_chunks - 1
        ):
            prev_chunk_count = self._job_counters[stage_id][prev_key]
            current_chunk_count = self._job_counters[stage_id][current_key]
            if prev_chunk_count <= current_chunk_count:
                return False

        prev_stage_job_end_time = self._get_dependency_job_end_time(
            job_type, chunk_id, stage_id, micro_batch_id
        )

        if prev_stage_job_end_time < 0:
            return False

        return True

    def _compare_forward_micro_batch_id(
        self, stage_id, last_stage_id, max_chunk_id
    ):
        for chunk_id in range(1, max_chunk_id + 1):
            if (
                self._job_counters[stage_id][f"forward{chunk_id}"]
                <= self._job_counters[last_stage_id][f"forward{chunk_id}"]
                < self.num_micro_batch
            ):
                return True
        return False

    def _get_stage_micro_batch_id(self, stage_id, job_type):
        micro_batch_id = 0
        for chunk_id in range(self.num_model_chunks):
            micro_batch_id += self._job_counters[stage_id][
                f"{job_type}{chunk_id}"
            ]
        return micro_batch_id

    def _fill_bubble_with_forward(
        self,
        chunk_ids,
        stage_id,
        next_job_start_time,
        insert_order="down",
    ):
        chunk_id = chunk_ids[0]
        while self._check_before_insert(
            "forward", chunk_ids, stage_id, next_job_start_time
        ):
            # After insert forward job, we need to check whether we can insert backward_b job
            if (
                self._stage_mem_usage[stage_id]
                + self.program_mem_usages[stage_id][f"forward{chunk_id}"]
                + self.program_max_mem_usages[stage_id][f"backward_b{chunk_id}"]
            ) > self.max_memory:
                break

            if insert_order == "down":
                stage_order = range(0, stage_id + 1)
            else:
                stage_order = range(self.num_stage - 1, stage_id - 1, -1)

            for stage_id in stage_order:
                if self._can_schedule_f_task(
                    stage_id, chunk_id
                ) and self._time_check(
                    "forward", chunk_id, stage_id, next_job_start_time
                ):
                    self._put_job_into_schedule("forward", chunk_id, stage_id)

            chunk_id = (chunk_id + 1) % len(chunk_ids)

    def _check_before_insert(
        self, job_type, chunk_ids, stage_id, next_job_start_time
    ):
        micro_batch_id_check = False
        for chunk_id in chunk_ids:
            micro_batch_id_check |= self._micro_batch_id_check(
                job_type, chunk_id, stage_id
            )

        return (
            self._memory_check(job_type, chunk_id, stage_id)
            and self._time_check(
                job_type, chunk_id, stage_id, next_job_start_time
            )
            and micro_batch_id_check
        )

    def _memory_check(self, job_type, chunk_id, stage_id):
        if (
            self._stage_mem_usage[stage_id]
            + self.program_max_mem_usages[stage_id][f"{job_type}{chunk_id}"]
            > self.max_memory
        ):
            return False
        return True

    def _time_check(self, job_type, chunk_id, stage_id, next_job_start_time):
        dependency_job_end_time = self._get_dependency_job_end_time(
            job_type,
            chunk_id,
            stage_id,
            self._job_counters[stage_id][f"{job_type}{chunk_id}"],
        )
        job_end_time = (
            max(
                self._stage_current_time[stage_id],
                dependency_job_end_time + self.program_runtime["communication"],
            )
            + self.program_runtime[job_type]
        )

        if job_end_time > next_job_start_time:
            return False
        return True

    def _micro_batch_id_check(self, job_type, chunk_id, stage_id):
        if (
            self._job_counters[stage_id][f"{job_type}{chunk_id}"]
            >= self.num_micro_batch
        ):
            return False
        return True

    def _put_job_into_schedule(
        self,
        job_type,
        chunk_id,
        stage_id,
    ):
        task_end_time = (
            self._stage_current_time[stage_id] + self.program_runtime[job_type]
        )

        micro_batch_id = self._job_counters[stage_id][f"{job_type}{chunk_id}"]
        if micro_batch_id >= self.num_micro_batch:
            raise ValueError(
                f"Job {job_type}{chunk_id} exceeds the limit of micro batches."
            )

        if (
            self._stage_mem_usage[stage_id]
            + self.program_max_mem_usages[stage_id][f"{job_type}{chunk_id}"]
            > self.max_memory
        ):
            raise ValueError(
                f"Job {job_type}{chunk_id} exceeds the memory limit."
            )

        self._check_job_chunk_order(
            job_type, chunk_id, stage_id, micro_batch_id
        )

        if job_type in ["forward", "backward_b"]:
            dependency_job_end_time = self._get_dependency_job_end_time(
                job_type, chunk_id, stage_id, micro_batch_id
            )
            if dependency_job_end_time < 0:
                prev_stage_id = self._get_prev_stage_id(
                    job_type, chunk_id, stage_id
                )
                raise ValueError(
                    f"Job {job_type}{chunk_id}_{micro_batch_id} at stage {stage_id} depends on unfinished job {job_type}{chunk_id}_{micro_batch_id} at stage {prev_stage_id}."
                )
            task_end_time = max(
                task_end_time,
                dependency_job_end_time
                + self.program_runtime["communication"]
                + self.program_runtime[job_type],
            )

        job_id = self._get_job_id(job_type, chunk_id, stage_id, micro_batch_id)
        if self._job_counters[stage_id]["forward0"] > 0:
            self._stage_bubbles[stage_id] += (
                task_end_time
                - self._stage_current_time[stage_id]
                - self.program_runtime[job_type]
            )

        self._job_end_times[job_id] = task_end_time
        self._stage_current_time[stage_id] = task_end_time
        self._stage_mem_usage[stage_id] += self.program_mem_usages[stage_id][
            f"{job_type}{chunk_id}"
        ]

        job_info = {
            "type": job_type,
            "chunk": chunk_id,
            "micro_batch": micro_batch_id,
        }
        self._stage_job_schedule[stage_id].append(job_info)
        if job_type == "backward_b":
            self._pending_w[stage_id].append((chunk_id, micro_batch_id))
        self._job_counters[stage_id][f"{job_type}{chunk_id}"] += 1

    def _put_w_job_into_schedule(self, stage_id):
        if not len(self._pending_w[stage_id]):
            raise ValueError("No pending backward_w job.")

        chunk_id, _ = self._pending_w[stage_id].popleft()
        self._put_job_into_schedule("backward_w", chunk_id, stage_id)

    def _check_job_chunk_order(
        self, job_type, chunk_id, stage_id, micro_batch_id
    ):
        if job_type == "forward":
            if chunk_id > 0:
                prev_job_end_time = self._job_end_times[
                    self._get_job_id(
                        "forward", chunk_id - 1, stage_id, micro_batch_id
                    )
                ]
                if prev_job_end_time < 0:
                    raise ValueError(
                        f"Job {job_type}{chunk_id}_{micro_batch_id} depends on unfinished {job_type}{chunk_id - 1}_{micro_batch_id} job."
                    )
        elif job_type == "backward_b":
            if chunk_id < self.num_model_chunks - 1:
                prev_job_end_time = self._job_end_times[
                    self._get_job_id(
                        job_type, chunk_id + 1, stage_id, micro_batch_id
                    )
                ]
                if prev_job_end_time < 0:
                    raise ValueError(
                        f"Job {job_type}{chunk_id}_{micro_batch_id} depends on unfinished {job_type}{chunk_id + 1}_{micro_batch_id} job."
                    )
        elif job_type == "backward_w":
            prev_job_id = self._get_job_id(
                "backward_b", chunk_id, stage_id, micro_batch_id
            )
            if self._job_end_times[prev_job_id] < 0:
                raise ValueError(
                    f"Job {job_type}{chunk_id}_{micro_batch_id} at stage {stage_id} depends on unfinished backward_b{chunk_id}_{micro_batch_id} job."
                )

    def _get_dependency_job_end_time(
        self, job_type, chunk_id, stage_id, micro_batch_id
    ):
        prev_stage_id = self._get_prev_stage_id(job_type, chunk_id, stage_id)
        if prev_stage_id < 0 or prev_stage_id >= self.num_stage:
            return 0

        prev_stage_job_id = self._get_job_id(
            job_type, chunk_id, prev_stage_id, micro_batch_id
        )

        prev_job_end_time = self._job_end_times[prev_stage_job_id]
        return prev_job_end_time

    def _get_prev_job_end_time(
        self, job_type, chunk_id, stage_id, micro_batch_id
    ):
        if job_type == "forward":
            prev_job_chunk_id = chunk_id - 1
            if prev_job_chunk_id < 0:
                return 0

        elif job_type == "backward_b":
            prev_job_chunk_id = chunk_id + 1
            if prev_job_chunk_id >= self.num_model_chunks:
                job_type = "forward"
                prev_job_chunk_id = self.num_model_chunks - 1

        prev_job_id = self._get_job_id(
            job_type, prev_job_chunk_id, stage_id, micro_batch_id
        )

        return self._job_end_times[prev_job_id]

    def _get_prev_stage_id(self, job_type, chunk_id, stage_id):
        if job_type == "forward":
            if chunk_id % 2:
                return stage_id + 1
            else:
                return stage_id - 1
        elif job_type in ["backward_b", "backward_w"]:
            if chunk_id % 2:
                return stage_id - 1
            else:
                return stage_id + 1

    def _get_max_stage_bubble(self, stage_id=-1, approved_bubbles=None):
        max_stage_bubble = max(self._stage_bubbles)
        if stage_id >= 0:
            max_approved_bubble = max(approved_bubbles)
            max_stage_bubble = max(
                max_stage_bubble,
                max_approved_bubble - approved_bubbles[stage_id],
            )
        return max_stage_bubble

    def _get_job_id(self, job_type, chunk_id, stage_id, job_micro_id):
        return (
            self.job_types.index(job_type)
            * self.num_model_chunks
            * self.num_stage
            * self.num_micro_batch
            + chunk_id * self.num_stage * self.num_micro_batch
            + stage_id * self.num_micro_batch
            + job_micro_id
        )

    def _get_bubble_rate(self):
        max_bubble = self._get_max_stage_bubble()
        fbw_cost = (
            self.program_runtime["forward"]
            + self.program_runtime["backward_w"]
            + self.program_runtime["communication"]
        )
        expected_time = fbw_cost * self.num_micro_batch * self.num_model_chunks
        bubble_rate = max_bubble / expected_time
        return bubble_rate
