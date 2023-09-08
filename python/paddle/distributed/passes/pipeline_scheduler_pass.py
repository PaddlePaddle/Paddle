# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.base import core
from paddle.distributed.auto_parallel.static.cost import calc_time_by_cost_model

from ..utils.log_utils import get_logger
from .pass_base import PassContext, new_pass, register_pass
from .pass_utils import (
    AutoParallelStreamType,
    _program_for_fthenb_and_1f1b,
    split_program,
)
from .pipeline_pass_base import PipelinePassBase

__not_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]

LR = "lr"
FORWARD = "forward"
BACKWARD = "backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_FThenB")
class PipelineFThenBPass(PipelinePassBase):
    def __init__(self):
        super().__init__()

    def _create_job_list(self):
        num_micro_batches = self.get_attr("num_micro_batches")

        job_list = []
        lr_job = core.Job(LR)
        job_list.append(lr_job)

        for i in range(num_micro_batches):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(i)
            job_list.append(forward_job)

        for i in range(num_micro_batches):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(i)
            job_list.append(backward_job)

        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        types = [LR, FORWARD, BACKWARD, OPT]
        sub_program_list = _program_for_fthenb_and_1f1b(program)
        return types, sub_program_list


@register_pass("pipeline_scheduler_1F1B")
class Pipeline1F1BPass(PipelinePassBase):
    def __init__(self):
        super().__init__()
        self.jobs_in_stable_phase = [BACKWARD, FORWARD]
        self.set_attr("enable_backward_forward_overlap", 0)

    # Backward-forward overlapping splits and rearranges jobs for pattern Bi-Fj.
    # For example: jobs = {..., BACKWARD-i, FORWARD-j, ...}, i < j
    # BACKWARD-i: Calc1 - Comm1 - Calc2 - Comm2 - Calc3
    # FORWARD-j: Calc4 - Comm3 - Calc5 - Comm4 - Calc6
    # Timeline:
    # ===Calc1==Comm1==Calc2==Comm2==Calc3==Calc4==Comm3==Calc5==Comm4==Calc6===
    #
    # After backward-forward overlapping: jobs = {Calc1, Comm1, Calc4, Comm3, Calc2, Comm2, Calc5, Comm4, Calc3, Calc6}
    # Timeline:
    # ===Calc1==Calc4==Calc2==Calc5==Calc3=Calc6===
    #         \       /     \       /
    #          \     /       \     /
    # ==========Comm1==Comm3==Comm2==Comm4==========
    #
    def _backward_forward_overlap(self, backward_program, forward_program):
        logger.info("Backward forward overlap enabled in 1F1B.")
        print(f"backward_program :: {backward_program}")
        print(f"fowr_program :: {forward_program}")
        # Split BACKWARD
        valid_comm_op_ids = [
            op_id
            for op_id, op in enumerate(backward_program.global_block().ops)
            if self.is_comm_op_valid_to_overlap(op)
        ]
        # TODO(Ruibiao): Constrain the number of valid comm ops to resolve the potential memory explosion issue.
        is_backward_split_point = (
            lambda program, op_id: op_id - 1 in valid_comm_op_ids
        )
        (
            splitted_backward_job_types,
            splitted_backward_programs,
        ) = self._split_program_for_overlapping(
            BACKWARD, backward_program, is_backward_split_point
        )
        self._multistreaming_for_overlapping(splitted_backward_programs)

        # Split FORWARD
        ops = forward_program.global_block().ops
        num_ops = len(ops)
        splitted_op_ids = []
        op_id = 0
        for splitted_backward_program in splitted_backward_programs:
            backward_op_to_overlap = (
                splitted_backward_program.global_block().ops[-1]
            )
            backward_cost_to_overlap = self._op_cost(backward_op_to_overlap)

            forward_cost_to_overlap = self._op_cost(ops[op_id])
            print(
                f"backward_op_to_overlap : {backward_op_to_overlap}, cost = {backward_cost_to_overlap}"
            )
            print(
                f"forward_op_to_overlap : {ops[op_id]}, cost = {forward_cost_to_overlap}"
            )

            while (
                op_id < num_ops
                and forward_cost_to_overlap <= backward_cost_to_overlap
            ):
                op_id += 1
                op = ops[op_id]
                # Force split when meet comm op since it cannot overlap with comm op in backward.
                if op_id > 0 and self.is_comm_op_valid_to_overlap(
                    ops[op_id - 1]
                ):
                    break

                print(
                    f"forward_op_to_overlap : {ops[op_id]}, cost = {self._op_cost(ops[op_id])}"
                )
                forward_cost_to_overlap += self._op_cost(ops[op_id])

            splitted_op_ids.append(op_id)
            if op_id >= num_ops:
                break

        is_forward_split_point = lambda program, op_id: op_id in splitted_op_ids
        (
            splitted_forward_job_types,
            splitted_forward_programs,
        ) = self._split_program_for_overlapping(
            FORWARD, forward_program, is_forward_split_point
        )
        self._multistreaming_for_overlapping(splitted_forward_programs)

        # Rearrange splitted chunks for BACKWARD and FORWARD
        self.jobs_in_stable_phase.clear()
        num_splitted_forward_jobs = len(splitted_forward_job_types)
        num_splitted_backward_jobs = len(splitted_backward_job_types)
        for idx in range(
            max(num_splitted_forward_jobs, num_splitted_backward_jobs)
        ):
            if idx < num_splitted_backward_jobs:
                self.jobs_in_stable_phase.append(
                    splitted_backward_job_types[idx]
                )
            if idx < num_splitted_forward_jobs:
                self.jobs_in_stable_phase.append(
                    splitted_forward_job_types[idx]
                )

        return (
            splitted_backward_job_types,
            splitted_backward_programs,
            splitted_forward_job_types,
            splitted_forward_programs,
        )

    def _create_job_list(self):
        num_micro_batches = self.get_attr("num_micro_batches")
        pp_stage = self.get_attr("pp_stage")
        pp_degree = self.get_attr("pp_degree")

        job_list = []
        lr_job = core.Job(LR)
        job_list.append(lr_job)

        assert (
            pp_degree <= num_micro_batches
        ), "Num of micro batches should larger than or equal to pp degree."

        micro_batch_in_warmup = pp_degree - pp_stage
        micro_batch_in_1f1b = num_micro_batches - micro_batch_in_warmup

        forward_micro_batch_id = 0
        for i in range(micro_batch_in_warmup):
            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        backward_micro_batch_id = 0
        for i in range(micro_batch_in_1f1b):
            for job_type in self.jobs_in_stable_phase:
                job = core.Job(job_type)
                micro_batch_id = (
                    forward_micro_batch_id
                    if job_type.startswith(FORWARD)
                    else backward_micro_batch_id
                )
                job.set_micro_batch_id(micro_batch_id)
                job_list.append(job)
            forward_micro_batch_id += 1
            backward_micro_batch_id += 1

        for i in range(micro_batch_in_warmup):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1

        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _multistreaming_for_overlapping(self, programs):
        # TODO(Ruibiao): Add cross-program event dependency for multi-stream.
        for program in programs:
            last_op = program.global_block().ops[-1]
            if self.is_comm_op_valid_to_overlap(last_op):
                last_op.dist_attr.execution_stream = (
                    AutoParallelStreamType.MP_STREAM.value
                )

    def _op_cost(self, op):
        try:
            return calc_time_by_cost_model(op)
        except:
            logger.info(f"The cost of {op} is unknown.")
            return 0.0

    def _partial_programs(self, program):
        types = [LR, FORWARD, BACKWARD, OPT]
        sub_programs = _program_for_fthenb_and_1f1b(program)

        enable_backward_forward_overlap = self.get_attr(
            "enable_backward_forward_overlap"
        )

        if enable_backward_forward_overlap:
            logger.info("Backward forward overlap enabled in 1F1B.")
            forward_program, backward_program = sub_programs[1], sub_programs[2]
            (
                splitted_backward_job_types,
                splitted_backward_programs,
                splitted_forward_job_types,
                splitted_forward_programs,
            ) = self._backward_forward_overlap(
                backward_program, forward_program
            )
            types += splitted_forward_job_types + splitted_backward_job_types
            sub_programs += (
                splitted_forward_programs + splitted_backward_programs
            )

        for i in range(len(types)):
            print(f"type = {types[i]}, sub_programs = {sub_programs[i]}\n")
        logger.info(f"jobs_in_stable_phase = {self.jobs_in_stable_phase}")

        return types, sub_programs

    def _split_program_for_overlapping(self, job_type, program, is_split_point):
        assert job_type in [
            FORWARD,
            BACKWARD,
        ], f"job_type should be one of {[FORWARD, BACKWARD]}"

        ops = program.global_block().ops
        num_ops = len(ops)

        split_ids = []
        for op_id in range(1, num_ops):
            if is_split_point(program, op_id):
                split_ids.append(op_id)

        splitted_programs, __, __ = split_program(program, split_ids)

        splitted_job_types = []
        num_splitted_programs = len(splitted_programs)
        for idx in range(num_splitted_programs):
            splitted_job_types.append(f"{job_type}(chunk{idx})")

        return splitted_job_types, splitted_programs

    def is_comm_op_valid_to_overlap(self, op):
        return (
            op.type == "c_allreduce_sum"
            and op.dist_attr.execution_stream
            == AutoParallelStreamType.CALC_STREAM.value
        )


def apply_pass(main_program, startup_program, pass_name, pass_attr={}):
    assert pass_name in [
        "FThenB",
        "1F1B",
    ], "pipeline scheduler only support FThenB and 1F1B, but recieve {}".format(
        pass_name
    )

    if pass_name == "1F1B":
        pass_attr["enable_backward_forward_overlap"] = int(
            os.environ.get("FLAGS_1f1b_backward_forward_overlap", 0)
        )

    pipeline_pass = new_pass("pipeline_scheduler_" + pass_name, pass_attr)
    pass_context = PassContext()
    pipeline_pass.apply([main_program], [startup_program], pass_context)
    plan = pass_context.get_attr("plan")
    return plan
