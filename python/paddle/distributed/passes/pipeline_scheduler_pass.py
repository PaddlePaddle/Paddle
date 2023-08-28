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

from paddle.fluid import core

from ..utils.log_utils import get_logger
from .pass_base import PassContext, new_pass, register_pass
from .pass_utils import _program_for_fthenb_and_1f1b, split_program
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
        # Backward-forward overlapping splits and rearranges jobs for pattern Bi-Fj.
        # For example: jobs = {..., BACKWARD-i, FORWARD-j, ...}, i < j
        # BACKWARD-i: OP1 - AllReduce - OP3
        # FORWARD-j: OP4 - AllReduce - OP6
        # Timeline:
        # ===OP1===AllReduce===OP2===OP3===AllReduce===OP4
        #
        # After backward-forward overlapping: jobs = {..., OP1, AllReduce, OP3, OP2, AllReduce, OP4}
        # Timeline:
        # === OP1 === OP3 =====OP2===========OP4
        #        \            /
        #         \          /
        # ========= AllReduce == AllReduce
        self.set_attr("num_comm_op_in_backward_forward_overlap", 0)

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
        job_list.append(opt_job)
        return job_list

    def _cost(self, op_type):
        cost = {
            "recv_v2": 0.229,
            "c_allreduce_sum": float(
                "INF"
            ),  # ONLY for Forward, set the cost of c_allreduce_sum as INF so all of them will be splitted to the end of a chunk.
            "cast": 0.052,
            "c_embedding": 0.061,
            "lookup_table_v2": 0.047,
            "elementwise_add": 0.051,
            "layer_norm": 0.086,
            "c_identity": 0.037,
            "matmul_v2": 0.660,
            "split": 0.070,
            "transpose2": 0.030,
            "scale": 0.019,
            "fused_softmax_mask_upper_triangle": 0.284,
            "gelu": 0.128,
        }
        return cost[op_type] if op_type in cost else 0.0

    def _multistreaming_for_overlapping(self, programs):
        for program in programs:
            last_op = program.global_block().ops[-1]
            if self.is_comm_op(last_op) and last_op.attr("use_calc_stream"):
                last_op.dist_attr.execution_stream = "allreduce_stream"

    def _partial_programs(self, program):
        types = [LR, FORWARD, BACKWARD, OPT]
        sub_programs = _program_for_fthenb_and_1f1b(program)

        num_comm_op_in_backward_forward_overlap = self.get_attr(
            "num_comm_op_in_backward_forward_overlap"
        )
        assert (
            num_comm_op_in_backward_forward_overlap >= 0
        ), f"Get num_comm_op_in_backward_forward_overlap = {num_comm_op_in_backward_forward_overlap}, which should be >= 0."

        if num_comm_op_in_backward_forward_overlap > 0:
            logger.info(
                f"Backward forward overlap enabled in 1F1B, num_comm_op_in_backward_forward_overlap = {num_comm_op_in_backward_forward_overlap}."
            )

            # Split FORWARD
            forward_program = sub_programs[1]
            ops = forward_program.global_block().ops
            num_ops = len(ops)

            costs = [self._cost(op.type) for op in ops]
            prefix_cost = 0
            duration_for_overlap = 0.771  # cost of allreduce in BACKWARD
            splitted_op_ids = []
            for op_id, op in enumerate(ops):
                if prefix_cost > duration_for_overlap:
                    splitted_op_ids.append(op_id)
                    prefix_cost = 0
                    if (
                        len(splitted_op_ids) + 1
                        >= num_comm_op_in_backward_forward_overlap
                    ):
                        break

                prefix_cost += self._cost(op.type)

            is_forward_split_point = (
                lambda program, op_id: op_id in splitted_op_ids
            )

            (
                splitted_forward_job_types,
                splitted_forward_programs,
            ) = self._split_program_for_overlapping(
                FORWARD, forward_program, is_forward_split_point
            )
            self._multistreaming_for_overlapping(splitted_forward_programs)
            types += splitted_forward_job_types
            sub_programs += splitted_forward_programs

            # Split BACKWARD
            backward_program = sub_programs[2]
            comm_op_ids = [
                op_id
                for op_id, op in enumerate(backward_program.global_block().ops)
                if self.is_comm_op(op)
            ]
            is_backward_split_point = (
                lambda program, op_id: op_id - 1 in comm_op_ids
                and len(comm_op_ids) - comm_op_ids.index(op_id - 1)
                < num_comm_op_in_backward_forward_overlap
            )
            (
                splitted_backward_job_types,
                splitted_backward_programs,
            ) = self._split_program_for_overlapping(
                BACKWARD, backward_program, is_backward_split_point
            )
            self._multistreaming_for_overlapping(splitted_backward_programs)
            types += splitted_backward_job_types
            sub_programs += splitted_backward_programs

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

    def is_comm_op(self, op):
        return op.type == "c_allreduce_sum"


def apply_pass(main_program, startup_program, pass_name, pass_attr={}):
    assert pass_name in [
        "FThenB",
        "1F1B",
    ], "pipeline scheduler only support FThenB and 1F1B, but recieve {}".format(
        pass_name
    )

    if pass_name == "1F1B":
        pass_attr["num_comm_op_in_backward_forward_overlap"] = int(
            os.environ.get("FLAGS_num_comm_op_in_backward_forward_overlap", 0)
        )

    pipeline_pass = new_pass("pipeline_scheduler_" + pass_name, pass_attr)
    pass_context = PassContext()
    pipeline_pass.apply([main_program], [startup_program], pass_context)
    plan = pass_context.get_attr("plan")
    return plan
