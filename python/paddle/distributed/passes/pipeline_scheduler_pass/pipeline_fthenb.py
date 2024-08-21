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
from ..pass_utils import (
    _program_for_fthenb_and_1f1b,
    _split_program_into_forward_backward_optimize,
)
from .pipeline_pass_base import PipelinePassBase

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
        # NOTE: The flag "enable_send_recv_overlap" may increase the reserved memory of GPUs.
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        types = [FORWARD, BACKWARD, OPT]
        sub_program_list = _program_for_fthenb_and_1f1b(
            program, enable_send_recv_overlap
        )
        return types, sub_program_list

    def _partial_pir_programs(self, program):
        # NOTE: The flag "enable_send_recv_overlap" may increase the reserved memory of GPUs.
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        types = [FORWARD, BACKWARD, OPT]
        sub_program_list = _split_program_into_forward_backward_optimize(
            program, enable_send_recv_overlap
        )
        return types, sub_program_list
