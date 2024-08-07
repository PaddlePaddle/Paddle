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

from paddle.base import core

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import _program_for_zero_bubble, split_matmul_grad_to_matmul
from .pipeline_pass_base import PipelinePassBase

FORWARD = "forward"
BACKWARD = "backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_ZBH1")
class PipelineZeroBubblePipelinePass(PipelinePassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("enable_optimizer_post_validation", 0)

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

    def _partial_programs(self, program):
        dist_context = self.get_attr("dist_context")
        self._split_matmul_grad_ops_to_matmul(program, dist_context)
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        types, sub_program_list = _program_for_zero_bubble(
            program, enable_send_recv_overlap
        )
        return types, sub_program_list
