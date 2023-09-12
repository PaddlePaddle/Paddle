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

from paddle.base import core

from .pass_base import PassContext, new_pass, register_pass
from .pass_utils import _program_for_fthenb_and_1f1b
from .pipeline_pass_base import PipelinePassBase

__not_shape_var_type__ = [
    core.VarDesc.VarType.READER,
    core.VarDesc.VarType.STEP_SCOPES,
    core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    core.VarDesc.VarType.FEED_MINIBATCH,
    core.VarDesc.VarType.FETCH_LIST,
]


@register_pass("pipeline_scheduler_FThenB")
class PipelineFThenBPass(PipelinePassBase):
    def __init__(self):
        super().__init__()

    def create_job_list(self):
        num_micro_batches = self.get_attr("num_micro_batches")

        job_list = []
        lr_job = core.Job("lr")
        job_list.append(lr_job)

        for i in range(num_micro_batches):
            forward_job = core.Job("forward")
            forward_job.set_micro_batch_id(i)
            job_list.append(forward_job)

        for i in range(num_micro_batches):
            backward_job = core.Job("backward")
            backward_job.set_micro_batch_id(i)
            job_list.append(backward_job)

        opt_job = core.Job("optimizer")
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def partial_programs(self, program):
        types = ["lr", "forward", "backward", "optimizer"]
        sub_program_list = _program_for_fthenb_and_1f1b(program)
        return types, sub_program_list


@register_pass("pipeline_scheduler_1F1B")
class Pipeline1F1BPass(PipelinePassBase):
    def __init__(self):
        super().__init__()

    def create_job_list(self):
        num_micro_batches = self.get_attr("num_micro_batches")
        pp_stage = self.get_attr("pp_stage")
        pp_degree = self.get_attr("pp_degree")

        job_list = []
        lr_job = core.Job("lr")
        job_list.append(lr_job)

        assert (
            pp_degree <= num_micro_batches
        ), "Num of micro batches should larger than pp degree."

        micro_batch_in_warmup = pp_degree - pp_stage
        micro_batch_in_1f1b = num_micro_batches - micro_batch_in_warmup

        forward_micro_batch_id = 0
        for i in range(micro_batch_in_warmup):
            forward_job = core.Job("forward")
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        backward_micro_batch_id = 0
        for i in range(micro_batch_in_1f1b):
            backward_job = core.Job("backward")
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1
            forward_job = core.Job("forward")
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        for i in range(micro_batch_in_warmup):
            backward_job = core.Job("backward")
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)
            backward_micro_batch_id += 1

        opt_job = core.Job("optimizer")
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def partial_programs(self, program):
        types = ["lr", "forward", "backward", "optimizer"]
        sub_program_list = _program_for_fthenb_and_1f1b(program)
        return types, sub_program_list


def apply_pass(main_program, startup_program, pass_name, pass_attr={}):
    assert pass_name in [
        "FThenB",
        "1F1B",
    ], "pipeline scheduler only support FThenB and 1F1B, but recieve {}".format(
        pass_name
    )
    pipeline_pass = new_pass("pipeline_scheduler_" + pass_name, pass_attr)
    pass_context = PassContext()
    pipeline_pass.apply([main_program], [startup_program], pass_context)
    plan = pass_context.get_attr("plan")
    return plan
