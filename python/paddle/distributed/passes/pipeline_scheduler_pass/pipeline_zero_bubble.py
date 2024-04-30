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
            "backward": 1000,
            "communication": 1,
        }

    def _create_job_list(self):
        pass

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
        max_memory,
    ):
        self.num_stage = num_stage
        self.num_micro_batch = num_micro_batch
        self.num_model_chunks = num_model_chunks
        self.num_nodes = num_model_chunks * num_stage * num_micro_batch
        self.program_mem_usages = program_mem_usages
        self.program_max_mem_usages = program_max_mem_usages
        self.job_types = ["forward", "backward_w", "backward_b"]
        self.max_memory = max_memory

    def init_schedule(self):
        job_counter = {}
        for job_type in self.job_types:
            for chunk_id in self.num_model_chunks:
                job_counter[f"{job_type}{chunk_id}"] = 0

        self._job_counters = [job_counter.copy() for _ in range(self.num_stage)]
        self._job_end_times = [-1] * self.num_nodes
        self._stage_current_time = [0] * self.num_stage
        self._stage_mem_usage = [0] * self.num_stage
        self._pending_w = [deque() for _ in range(self.num_stage)]
        self._stage_job_schedule = [[] for _ in range(self.num_stage)]
        self._stage_bubbles = [0] * self.num_stage

    def create_v_schedule(
        self, fill_f=True, fill_w=True, approved_bubbles=None
    ):
        self.init_schedule()
        if approved_bubbles is None:
            approved_bubbles = [-1] * self.num_stage
        max_approved_bubble = max(approved_bubbles)

        self._insert_warmup_stage_forward_jobs()
        self._insert_forward_jobs_before_backward_b()

    def _insert_warmup_stage_forward_jobs(self):
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

            self._fill_bubble("forward", [1], i, forward1_start_time)
            self._put_job_into_schedule("forward", chunk_id=1, stage_id=i)

    def _insert_forward_jobs_before_backward_b(self):
        # Insert left chunk forward jobs
        insert_order = "down"
        for i in range(2, self.num_model_chunks):
            if insert_order == "down":
                start_insert_stage_id = 0
                stage_order_list = list(range(1, self.num_stage))
            else:
                start_insert_stage_id = self.num_stage - 1
                stage_order_list = list(range(self.num_stage - 2, -1, -1))

            fill_chunk_ids = list(range(0, i))
            self._put_job_into_schedule(
                "forward", chunk_id=i, stage_id=start_insert_stage_id
            )

            for stage_id in stage_order_list:
                last_stage_id = (
                    stage_id - 1 if insert_order == "down" else stage_id + 1
                )
                next_chunk_forward_start_time = (
                    self._job_end_times[
                        self._get_job_id("forward", i, last_stage_id, 0)
                    ]
                    + self.program_runtime["communication"]
                )

                self._fill_bubble(
                    "forward",
                    fill_chunk_ids,
                    stage_id,
                    next_chunk_forward_start_time,
                    insert_order=insert_order,
                )

                self._put_job_into_schedule(
                    "forward", chunk_id=i, stage_id=stage_id
                )

            insert_order = "up" if insert_order == "down" else "down"

        # Insert forward jobs before backward_b
        if insert_order == "down":
            start_insert_stage_id = 0
            stage_order_list = list(range(1, self.num_stage))
        else:
            start_insert_stage_id = self.num_stage - 1
            stage_order_list = list(range(self.num_stage - 2, -1, -1))

        chunk_id = 0
        for stage_id in stage_order_list:
            last_stage_id = (
                stage_id - 1 if insert_order == "down" else stage_id + 1
            )

            # Each stage's forward job count should not be less than the last stage
            while (
                self._get_stage_job_cnt(stage_id, "forward")
                < self._get_stage_job_cnt(last_stage_id, "forward")
            ) or self._compare_forward_job_cnt(stage_id, last_stage_id):
                if insert_order == "down":
                    stage_order = range(stage_id, self.num_stage)
                else:
                    stage_order = range(stage_id, -1, -1)

                for id in stage_order:
                    if self._job_cnt_check("forward", chunk_id, id):
                        self._put_job_into_schedule("forward", chunk_id, id)
                chunk_id = (chunk_id + 1) % self.num_model_chunks

    def _compare_forward_job_cnt(self, stage_id, last_stage_id):
        for chunk_id in range(1, self.num_model_chunks):
            if (
                self._job_counters[stage_id][f"forward{chunk_id}"]
                <= self._job_counters[last_stage_id][f"forward{chunk_id}"]
                < self.num_micro_batch
            ):
                return True
        return False

    def _get_stage_job_cnt(self, stage_id, job_type):
        job_cnt = 0
        for chunk_id in range(self.num_model_chunks):
            job_cnt += self._job_counters[stage_id][f"{job_type}{chunk_id}"]
        return job_cnt

    def _fill_bubble(
        self,
        fill_job_type,
        chunk_ids,
        stage_id,
        next_job_start_time,
        insert_order="down",
    ):
        chunk_id = chunk_ids[0]
        while self._check_before_insert(
            fill_job_type, chunk_id, stage_id, next_job_start_time
        ):
            # After insert forward job, we need to check whether we can insert backward_w job
            if fill_job_type == "forward":
                if (
                    self._stage_mem_usage[stage_id]
                    + self.program_mem_usages[f"{fill_job_type}{chunk_id}"]
                    + self.program_max_mem_usages[f"backward_w{chunk_id}"]
                ) > self.max_memory:
                    break

            if insert_order == "down":
                stage_order = range(stage_id, self.num_stage)
            else:
                stage_order = range(stage_id, -1, -1)

            for stage_id in stage_order:
                if self._job_cnt_check(fill_job_type, chunk_id, stage_id):
                    self._put_job_into_schedule(
                        fill_job_type, chunk_id, stage_id
                    )

            chunk_id = (chunk_id + 1) % len(chunk_ids)

    def _check_before_insert(
        self, job_type, chunk_id, stage_id, next_job_start_time
    ):
        return (
            self._memory_check(job_type, chunk_id, stage_id)
            and self._time_check(
                job_type, chunk_id, stage_id, next_job_start_time
            )
            and self._job_cnt_check(job_type, chunk_id, stage_id)
        )

    def _memory_check(self, job_type, chunk_id, stage_id):
        if (
            self._stage_mem_usage[stage_id]
            + self.program_mem_usages[f"{job_type}{chunk_id}"]
            > self.max_memory
        ):
            return False
        return True

    def _time_check(self, job_type, chunk_id, stage_id, next_job_start_time):
        if (
            self._stage_current_time[stage_id] + self.program_runtime[job_type]
            > next_job_start_time
        ):
            return False
        return True

    def _job_cnt_check(self, job_type, chunk_id, stage_id):
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

        job_cnt = self._job_counters[stage_id][f"{job_type}{chunk_id}"]
        if job_cnt > self.num_micro_batch:
            raise ValueError(
                f"Job {job_type}{chunk_id} exceeds the limit of micro batches."
            )

        if (
            self._stage_mem_usage[stage_id]
            + self.program_max_mem_usages[f"{job_type}{chunk_id}"]
            > self.max_memory
        ):
            raise ValueError(
                f"Job {job_type}{chunk_id} exceeds the memory limit."
            )

        if chunk_id > 0:
            self._check_job_dependency_finished(
                job_type, chunk_id, stage_id, job_cnt
            )

        if chunk_id > 0 and job_type != "optimize":
            if stage_id < self.num_stage - 1:
                next_stage_job_id = self._get_job_id(
                    job_type, chunk_id, stage_id + 1, job_cnt
                )
                job_end_time = self._job_end_times[next_stage_job_id]
                if job_end_time < 0:
                    raise ValueError(
                        f"Job {job_type}{chunk_id} in stage {stage_id} depends on unfinished job in stage {stage_id + 1}."
                    )
                task_end_time = max(
                    task_end_time,
                    job_end_time
                    + self.program_runtime["communication"]
                    + self.program_runtime[job_type],
                )

        if chunk_id == 0 and job_type != "optimize":
            if stage_id > 0:
                prev_stage_job_id = self._get_job_id(
                    job_type, chunk_id, stage_id - 1, job_cnt
                )
                job_end_time = self._job_end_times[prev_stage_job_id]
                if job_end_time < 0:
                    raise ValueError(
                        f"Job {job_type}{chunk_id} in stage {stage_id} depends on unfinished job in stage {stage_id - 1}."
                    )
                task_end_time = max(
                    task_end_time,
                    job_end_time
                    + self.program_runtime["communication"]
                    + self.program_runtime[job_type],
                )

        job_id = self._get_job_id(job_type, chunk_id, stage_id, job_cnt)
        if self._job_counters[stage_id]["forward0"] > 0:
            self._stage_bubbles += (
                task_end_time
                - self._stage_current_time[stage_id]
                - self.program_runtime[job_type]
            )

        self._job_end_times[job_id] = task_end_time
        self._stage_current_time[stage_id] = task_end_time
        self._stage_mem_usage[stage_id] += self.program_mem_usages[
            f"{job_type}{chunk_id}"
        ]

        self._stage_job_schedule[stage_id].append(f"{job_type}{chunk_id}")
        if job_type == "backward_b":
            self._pending_w[stage_id].append((chunk_id, job_cnt))
        self._job_counters[stage_id][f"{job_type}{chunk_id}"] += 1

    def _put_w_job_into_schedule(self, stage_id):
        if not len(self._pending_w[stage_id]):
            raise ValueError("No pending backward_w job.")

        chunk_id, _ = self._pending_w[stage_id].popleft()
        self._put_job_into_schedule("backward_w", chunk_id, stage_id)

    def _check_job_dependency_finished(
        self, job_type, chunk_id, stage_id, job_cnt
    ):
        if job_type in ["backward_b", "backward_w"]:
            prev_job_end_time = self._job_end_times[
                self._get_job_id("backward", chunk_id - 1, stage_id, job_cnt)
            ]
            if prev_job_end_time < 0:
                raise ValueError(
                    f"Job {job_type}{chunk_id} depends on unfinished backward job."
                )
        elif job_type == "optimize":
            prev_job_end_time = self._job_end_times[
                self._get_job_id("backward_w", chunk_id, stage_id, job_cnt)
            ]
            if prev_job_end_time < 0:
                raise ValueError(
                    f"Job {job_type}{chunk_id} depends on unfinished backward job."
                )

    def _get_max_stage_bubble(
        self, stage_bubbles, approved_bubbles, stage_id=-1
    ):
        max_stage_bubble = max(stage_bubbles)
        if stage_id >= 0:
            max_stage_bubble = max(
                max_stage_bubble, max_stage_bubble - approved_bubbles[stage_id]
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
