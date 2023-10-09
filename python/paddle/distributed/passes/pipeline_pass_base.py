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

from .pass_base import PassBase
from .pass_utils import set_skip_gc_vars


class PipelinePassBase(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _create_job_list(self):
        """
        An interface that MUST be implemented by subclasses.
        """
        pass

    def _partial_programs(self, program):
        """
        An interface that MUST be implemented by subclasses.
        The return value MUST be two lists, one is a list of types(str), another
        is a list of sub programs.
        For example:
        return [LR, FORWARD, BACKWARD, OPT], [lr_prog, fwd_prog, bwd_prog, opt_prog]
        or
        return [FORWARD], [fwd_prog]
        """
        pass

    def _apply_single_impl(self, main_program, startup_program, context):
        """
        The shared process is implemented in this function and new subclass only need
        to implement two interfaces above, 'create_job_list' and 'partial_programs'.
        """
        job_types, sub_programs = self._partial_programs(main_program)
        jobs = self._create_job_list()

        type_to_program = dict(zip(job_types, sub_programs))
        set_skip_gc_vars(
            self.get_attr("num_micro_batches"), type_to_program, jobs
        )

        for type in type_to_program.keys():
            type_to_program[type] = type_to_program[type].desc
        plan = core.Plan(jobs, type_to_program)
        context.set_attr("plan", plan)
