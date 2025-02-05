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
            backward_b_job = core.Job(BACKWARD + "_b")
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
            backward_b_job = core.Job(BACKWARD + "_b")
            backward_b_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_b_job)

            backward_w_job = core.Job(BACKWARD + "_w")
            backward_w_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_w_job)
        else:
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
        backward_micro_batch_id += 1

        for i in range(pp_stage):
            backward_w_job = core.Job(BACKWARD + "_w")
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
        self.set_attr("program_runtimes", [61, 72, 71, 34, 3])
        self.set_attr("memory_limit_times", -1)

        self.program_mem_usages = []
        self.program_max_mem_usages = []
        self.base_memory = []
        self.program_runtime = {}

    def _create_job_list(self):
        pp_degree = self.get_attr("pp_degree")
        num_micro_batches = self.get_attr("num_micro_batches")
        num_model_chunks = self.get_attr("vpp_degree")

        assert num_micro_batches % pp_degree == 0

        # TODO(luchang): Fix the gradient explosion issue when  num_model_chunks(accumulate steps) > pp_degree
        assert (
            num_micro_batches <= pp_degree
        ), "zbvpp now only supports accumulate steps <= pp degree. It will cause gradient exploitation when accumulate steps > pp degree."

        program_runtimes = self.get_attr("program_runtimes")

        self.program_runtime = {
            "forward": program_runtimes[0],
            "backward_b": program_runtimes[1],
            "backward_w": program_runtimes[2],
            "loss": program_runtimes[3],
            "communication": program_runtimes[4],
        }

        v_scheduler = VScheduleCreator(
            pp_degree,
            num_micro_batches,
            num_model_chunks,
            self.program_mem_usages,
            self.program_max_mem_usages,
            self.base_memory,
            self.program_runtime,
            self._get_max_memory(),
        )

        schedule, end_time = None, None
        for fill_w_before_b in [True, False]:
            for fill_w_before_f in [True, False]:
                for fill_loss_stage in [True, False]:
                    if schedule is None:
                        schedule, end_time, _ = v_scheduler.create_v_schedule(
                            fill_w_before_b=fill_w_before_b,
                            fill_w_before_f=fill_w_before_f,
                            fill_loss_stage=fill_loss_stage,
                        )
                    else:
                        (
                            new_schedule,
                            new_end_time,
                            _,
                        ) = v_scheduler.create_v_schedule(
                            fill_w_before_b=fill_w_before_b,
                            fill_w_before_f=fill_w_before_f,
                            fill_loss_stage=fill_loss_stage,
                        )
                        if max(new_end_time) < max(end_time):
                            schedule, end_time = new_schedule, new_end_time

        stage_schedule = schedule[self.get_attr("pp_stage")]
        job_list = []

        for job_info in stage_schedule:
            job = core.Job(f"{job_info['type']}{job_info['chunk']}")
            job.set_micro_batch_id(job_info["micro_batch"])
            job_list.append(job)

        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)

        return job_list

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        num_model_chunks = self.get_attr("vpp_degree")
        memory_limit_times = self.get_attr("memory_limit_times")

        self._split_matmul_grad_ops_to_matmul(program, dist_context)
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        types, sub_program_list = _program_for_zero_bubble_vpp(
            program, num_model_chunks, dist_context, enable_send_recv_overlap
        )

        rank = paddle.distributed.get_rank()
        pp_group = []
        for process_mesh in dist_context.process_meshes:
            if rank in process_mesh.process_ids:
                pp_idx = process_mesh.process_ids.index(rank)
                for process_mesh in dist_context.process_meshes:
                    pp_group.append(process_mesh.process_ids[pp_idx])
                break

        if memory_limit_times > 0:
            self._estimate_program_mem_usagess(
                types, sub_program_list, dist_context, pp_group
            )
            self._get_all_device_base_memory(pp_group)
        else:
            self.program_mem_usages = [
                {type: 0 for type in types} for _ in pp_group
            ]
            self.program_max_mem_usages = [
                {type: 0 for type in types} for _ in pp_group
            ]
            self.base_memory = [0 for _ in range(len(pp_group))]

        return types, sub_program_list

    def _estimate_program_mem_usagess(
        self, types, sub_program_list, dist_context, pp_group
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
        with paddle.base.dygraph.guard():
            all_mem_usages = []
            all_max_usages = []
            paddle.distributed.all_gather_object(all_mem_usages, mem_usages)
            paddle.distributed.all_gather_object(all_max_usages, max_mem_usages)

        self.program_mem_usages = [{} for _ in range(len(pp_group))]
        self.program_max_mem_usages = [{} for _ in range(len(pp_group))]

        for i, id in enumerate(pp_group):
            for j, type in enumerate(types):
                self.program_mem_usages[i][type] = all_mem_usages[id][j]
                self.program_max_mem_usages[i][type] = all_max_usages[id][j]

    def _get_all_device_base_memory(self, pp_group):
        with paddle.base.dygraph.guard():
            self.base_memory = []
            all_base_memory = []
            rank = paddle.distributed.get_rank()
            base_memory = paddle.device.cuda.memory_allocated(rank)
            paddle.distributed.all_gather_object(all_base_memory, base_memory)
            for id in pp_group:
                self.base_memory.append(all_base_memory[id])

    def _get_max_memory(self):
        memory_limit_times = self.get_attr("memory_limit_times")

        if memory_limit_times < 0:
            return float("inf")

        num_model_chunks = self.get_attr("vpp_degree")
        micro_batch_in_warmup = self.get_attr("pp_degree")
        base_memory = max(self.base_memory)

        forward_cost = 0
        for i in range(num_model_chunks):
            forward_cost += self.program_mem_usages[0][f"forward{i}"]

        backward_max_cost = 0
        backward_cost = 0
        for i in range(num_model_chunks):
            backward_max_cost = max(
                backward_max_cost,
                backward_cost
                + self.program_max_mem_usages[0][f"backward_b{i}"],
            )
            backward_cost += self.program_max_mem_usages[0][f"backward_b{i}"]

        memory_1f1b = base_memory + backward_max_cost
        for i in range(micro_batch_in_warmup):
            memory_1f1b += forward_cost

        return memory_1f1b * memory_limit_times


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
        self.calculate_loss_stage = (
            0 if num_model_chunks % 2 == 0 else num_stage - 1
        )

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
        self,
        fill_w_before_f=True,
        fill_w_before_b=True,
        fill_loss_stage=True,
        approved_bubbles=None,
    ):
        self.init_schedule()
        if approved_bubbles is None:
            approved_bubbles = [-1] * self.num_stage
        max_approved_bubble = max(approved_bubbles)

        self._insert_forward_jobs_before_forward1()
        self._insert_forward_jobs_before_backward_b()
        self._insert_jobs_after_backward_start(
            fill_w_before_f, fill_w_before_b, fill_loss_stage, approved_bubbles
        )

        schedule = self._stage_job_schedule.copy()
        end_time = self._job_end_times.copy()
        max_bubble = self._get_max_stage_bubble()

        if max_approved_bubble < 0 or max_bubble < max_approved_bubble:
            new_schedule, new_end_time, new_max_bubble = self.create_v_schedule(
                fill_w_before_f,
                fill_w_before_b,
                fill_loss_stage,
                self._stage_bubbles,
            )

            if max(new_end_time) < max(end_time):
                return new_schedule, new_end_time, new_max_bubble

        return schedule, end_time, max_bubble

    def _insert_forward_jobs_before_forward1(self):
        # Step1: Insert forward jobs with chunk_id=0 into the schedule
        for i in range(self.num_stage):
            self._put_job_into_schedule("forward", chunk_id=0, stage_id=i)

        # Step2: Insert forward jobs with chunk_id=1 into the schedule
        self._fill_forward_before_one_job(
            "forward", 1, [0], forward_insert_order="down"
        )

    def _insert_forward_jobs_before_backward_b(self):
        chunk_to_insert_order = {
            0: "down",
            1: "up",
        }

        # Insert the rest chunk_id forward jobs
        for chunk_id in range(2, self.num_model_chunks):
            fill_chunk_ids = list(range(0, chunk_id))
            forward_insert_order = chunk_to_insert_order[fill_chunk_ids[-1] % 2]
            self._fill_forward_before_one_job(
                "forward", chunk_id, fill_chunk_ids, forward_insert_order
            )

        # Insert forward jobs to fill the bubble before backward_b0
        fill_chunk_ids = list(range(0, self.num_model_chunks))
        forward_insert_order = chunk_to_insert_order[fill_chunk_ids[-1] % 2]

        self._fill_forward_before_one_job(
            "backward_b",
            0,
            list(range(0, self.num_model_chunks)),
            forward_insert_order,
            insert_end_point_job=False,
        )

    def _fill_forward_before_one_job(
        self,
        end_point_job_type,
        end_point_chunk_id,
        fill_chunk_ids,
        forward_insert_order,
        insert_end_point_job=True,
    ):
        stage_order = list(range(self.num_stage))
        if forward_insert_order == "down":
            stage_order.reverse()

        stage_last_job = self._stage_job_schedule[stage_order[0]][-1]
        end_point_job_start_time = self._job_end_times[
            self._get_job_id(
                stage_last_job["type"],
                stage_last_job["chunk"],
                stage_order[0],
                stage_last_job["micro_batch"],
            )
        ]

        if insert_end_point_job:
            self._put_job_into_schedule(
                end_point_job_type, end_point_chunk_id, stage_order[0]
            )

        for stage_id in stage_order[1:]:
            stage_last_job = self._stage_job_schedule[stage_id][-1]
            start_fill_chunk = (stage_last_job["chunk"] + 1) % len(
                fill_chunk_ids
            )

            end_point_job_start_time = (
                end_point_job_start_time
                + self.program_runtime["communication"]
                + self._get_program_runtime(
                    end_point_job_type, stage_id, end_point_chunk_id
                )
            )

            self._fill_bubble_with_forward(
                stage_id,
                fill_chunk_ids,
                start_fill_chunk,
                end_point_job_start_time,
                insert_order=forward_insert_order,
            )

            if insert_end_point_job:
                self._put_job_into_schedule(
                    end_point_job_type, end_point_chunk_id, stage_id
                )

    def _insert_jobs_after_backward_start(
        self,
        fill_w_before_f,
        fill_w_before_b,
        fill_loss_stage,
        approved_bubbles,
    ):
        backward_b_job_number = self.num_model_chunks * self.num_micro_batch

        first_backward_b_stage = (
            0 if self.num_model_chunks % 2 == 0 else self.num_stage - 1
        )
        while (
            self._get_stage_backward_b_number(first_backward_b_stage)
            < backward_b_job_number
        ):
            # Step1: Check memory usage, if not enough, put pending backward_w job into schedule
            for stage_id in range(self.num_stage):
                while not self._memory_check("backward_b", 0, stage_id):
                    if len(self._pending_w[stage_id]) == 0:
                        raise ValueError(
                            f"No pending backward_w job and backward_b0 job exceeds the memory limit at stage {stage_id}."
                        )
                    self._put_w_job_into_schedule(stage_id)

            # Step2: Insert backward_b job for each stage
            # b_ranks = [[] for _ in range(self.num_model_chunks)]
            backward_insert_order = range(self.num_stage)
            if self.num_model_chunks % 2:
                backward_insert_order = range(self.num_stage - 1, -1, -1)

            for stage_id in backward_insert_order:
                for chunk_id in range(0, self.num_model_chunks):
                    if self._can_schedule_b_task(stage_id, chunk_id):
                        dependency_job_end_time = (
                            self._get_dependency_job_end_time(
                                "backward_b",
                                chunk_id,
                                stage_id,
                                self._job_counters[stage_id][
                                    f"backward_b{chunk_id}"
                                ],
                            )
                        )

                        while len(
                            self._pending_w[stage_id]
                        ) and dependency_job_end_time + self.program_runtime[
                            "communication"
                        ] >= self._stage_current_time[
                            stage_id
                        ] + self._get_program_runtime(
                            "backward_w",
                            stage_id,
                            self._pending_w[stage_id][0][0],
                        ):
                            self._put_w_job_into_schedule(stage_id)

                        if (
                            stage_id == self.calculate_loss_stage
                            and fill_loss_stage
                        ):
                            while (
                                len(self._pending_w[stage_id])
                                and dependency_job_end_time
                                + self.program_runtime["communication"]
                                >= self._stage_current_time[stage_id]
                                + self._get_program_runtime(
                                    "backward_w",
                                    stage_id,
                                    self._pending_w[stage_id][0][0],
                                )
                                * 0.2
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
                            if fill_w_before_b:
                                self._put_w_job_into_schedule(stage_id)

                        self._put_job_into_schedule(
                            "backward_b", chunk_id, stage_id
                        )
                        break

            # Step3: Insert forward jobs after backward_b
            forward_insert_order = range(self.num_stage)
            if self.num_model_chunks % 2:
                forward_insert_order = range(self.num_stage - 1, -1, -1)

            for stage_id in forward_insert_order:
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
                        while len(
                            self._pending_w[stage_id]
                        ) and dependency_job_end_time + self.program_runtime[
                            "communication"
                        ] >= self._stage_current_time[
                            stage_id
                        ] + self._get_program_runtime(
                            "backward_w",
                            stage_id,
                            self._pending_w[stage_id][0][0],
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
        return self._can_schedule_task("backward_b", chunk_id, stage_id)

    def _can_schedule_task(self, job_type, chunk_id, stage_id):
        if job_type == "forward":
            current_key = f"forward{chunk_id}"
            prev_key = f"forward{chunk_id - 1}"
        elif job_type == "backward_b":
            current_key = f"backward_b{chunk_id}"
            prev_chunk_id = chunk_id + 1
            if prev_chunk_id >= self.num_model_chunks:
                prev_key = f"forward{self.num_model_chunks - 1}"
            else:
                prev_key = f"backward_b{chunk_id + 1}"

        micro_batch_id = self._job_counters[stage_id][current_key]
        if micro_batch_id >= self.num_micro_batch:
            return False

        if (job_type == "forward" and chunk_id > 0) or (
            job_type == "backward_b"
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

    def _get_stage_micro_batch_id(self, stage_id, job_type):
        micro_batch_id = 0
        for chunk_id in range(self.num_model_chunks):
            micro_batch_id += self._job_counters[stage_id][
                f"{job_type}{chunk_id}"
            ]
        return micro_batch_id

    def _fill_bubble_with_forward(
        self,
        stage_id,
        chunk_ids,
        start_chunk_id,
        next_job_start_time,
        insert_order="down",
    ):
        chunk_id = start_chunk_id
        less_forward_number = 0
        stage_order = range(0, stage_id + 1)
        if insert_order == "up":
            less_forward_number = self._get_stage_forward_number(0, chunk_ids)
            stage_order = range(self.num_stage - 1, stage_id - 1, -1)

        while self._check_before_insert(
            chunk_ids, stage_order, next_job_start_time, less_forward_number
        ):
            available_memory = self.max_memory - self._stage_mem_usage[stage_id]
            # After insert forward job, we need to check whether we can insert backward_b job
            available_memory -= self.program_max_mem_usages[stage_id][
                f"backward_b{chunk_id}"
            ]

            # Check whether we can insert all chunk_id forward jobs
            for i in range(1, self.num_model_chunks):
                if self._job_counters[stage_id][f"forward{i}"] == 0:
                    available_memory -= self.program_max_mem_usages[stage_id][
                        f"forward{i}"
                    ]
            if (
                available_memory
                < self.program_max_mem_usages[stage_id][f"forward{chunk_id}"]
            ):
                break

            for i in stage_order:
                if self._can_schedule_f_task(i, chunk_id):
                    stage_forward_number = self._get_stage_forward_number(
                        i, chunk_ids
                    )
                    if stage_forward_number >= less_forward_number:
                        if not self._time_check(
                            "forward", chunk_id, i, next_job_start_time
                        ):
                            continue
                    self._put_job_into_schedule("forward", chunk_id, i)

            chunk_id = (chunk_id + 1) % len(chunk_ids)

    def _check_before_insert(
        self, chunk_ids, stage_order, next_job_start_time, less_forward_number
    ):
        stage_id = stage_order[-1]
        if (
            self._get_stage_forward_number(stage_id, chunk_ids)
            < less_forward_number
        ):
            return True

        job_numbers = []
        for chunk_id in chunk_ids:
            job_numbers.append(
                self._job_counters[stage_id][f"forward{chunk_id}"]
            )

        micro_batch_id_check = False
        for number in job_numbers:
            if number < self.num_micro_batch:
                micro_batch_id_check = True
                break

        can_insert = False
        for chunk_id in chunk_ids:
            for i in stage_order:
                if self._can_schedule_f_task(i, chunk_id):
                    stage_forward_number = self._get_stage_forward_number(
                        i, chunk_ids
                    )
                    if stage_forward_number >= less_forward_number:
                        if not self._time_check(
                            "forward", chunk_id, i, next_job_start_time
                        ):
                            continue
                    can_insert = True
                    break

        return (
            self._memory_check("forward", chunk_id, stage_id)
            and micro_batch_id_check
            and can_insert
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
        job_end_time = max(
            self._stage_current_time[stage_id],
            dependency_job_end_time + self.program_runtime["communication"],
        ) + self._get_program_runtime(job_type, stage_id, chunk_id)

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
        program_runtime = self._get_program_runtime(
            job_type, stage_id, chunk_id
        )
        task_end_time = self._stage_current_time[stage_id] + program_runtime

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
                f"Job {job_type}{chunk_id} exceeds the memory limit at stage {stage_id}."
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
                    f"Job {job_type}{chunk_id}_{micro_batch_id} at stage {stage_id} depends on unscheduled job {job_type}{chunk_id}_{micro_batch_id} at stage {prev_stage_id}."
                )
            task_end_time = max(
                task_end_time,
                dependency_job_end_time
                + self.program_runtime["communication"]
                + program_runtime,
            )

        job_id = self._get_job_id(job_type, chunk_id, stage_id, micro_batch_id)
        if self._job_counters[stage_id]["forward0"] > 0:
            self._stage_bubbles[stage_id] += (
                task_end_time
                - self._stage_current_time[stage_id]
                - program_runtime
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

    def _get_stage_forward_number(self, stage, chunk_ids=None):
        job_number = 0
        if chunk_ids is None:
            chunk_ids = range(self.num_model_chunks)
        for chunk_id in chunk_ids:
            job_number += self._job_counters[stage][f"forward{chunk_id}"]
        return job_number

    def _get_stage_backward_b_number(self, stage, chunk_ids=None):
        job_number = 0
        if chunk_ids is None:
            chunk_ids = range(self.num_model_chunks)
        for chunk_id in chunk_ids:
            job_number += self._job_counters[stage][f"backward_b{chunk_id}"]
        return job_number

    def _get_program_runtime(self, job_type, stage_id, chunk_id):
        program_runtime = self.program_runtime[job_type]

        if job_type == "communication":
            return program_runtime

        if (
            stage_id == self.calculate_loss_stage
            and chunk_id == self.num_model_chunks - 1
        ):
            program_runtime += self.program_runtime["loss"]
        return program_runtime
