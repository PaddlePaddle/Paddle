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

        mem_usages = paddle.to_tensor(mem_usages)
        max_mem_usages = paddle.to_tensor(max_mem_usages)

        # Get program memory usage from all devices
        all_mem_usages = []
        all_max_usages = []
        paddle.distributed.all_gather_object(all_mem_usages, mem_usages)
        paddle.distributed.all_gather_object(all_max_usages, max_mem_usages)

        self.program_mem_usages = [{} for _ in range(len(all_mem_usages))]
        self.program_max_mem_usages = [{} for _ in range(len(all_max_usages))]

        for id in range(len(all_mem_usages)):
            for i, type in enumerate(types):
                self.program_mem_usages[id][type] = all_mem_usages[id][i]
                self.program_max_mem_usages[id][type] = all_max_usages[id][i]

    def _get_all_device_base_memory(self):
        self.base_memory = []
        rank = self.get_attr("pp_stage")
        base_memory = paddle.device.cuda.memory_allocated(rank)
        base_memory = paddle.to_tensor(base_memory)
        paddle.distributed.all_gather(self.base_memory, base_memory)
        self.base_memory = []
