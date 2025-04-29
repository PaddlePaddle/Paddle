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

import paddle
from paddle.base import core

from ...utils.log_utils import get_logger
from ..pass_base import PassBase
from ..pass_utils import (
    set_skip_gc_vars,
)

logger = get_logger(logging.INFO)


class PipelinePassBase(PassBase):

    # Pipeline stages
    RECV_FORWARD = "recv_forward"
    SEND_BACKWARD = "send_backward"
    FORWARD = "forward"
    BACKWARD = "backward"
    OPT = "optimizer"

    def __init__(self):
        super().__init__()
        self._in_pir_mode = paddle.base.framework.get_flags(
            "FLAGS_enable_pir_api"
        )["FLAGS_enable_pir_api"]

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
        return [FORWARD, BACKWARD, OPT], [fwd_prog, bwd_prog, opt_prog]
        or
        return [FORWARD], [fwd_prog]
        """
        pass

    def _apply_impl(self, main_programs, startup_programs, context):
        for main_program, startup_program in zip(
            main_programs, startup_programs
        ):
            if self._in_pir_mode:
                self._apply_pir_single_impl(
                    main_program, startup_program, context
                )
            else:
                raise NotImplementedError(
                    "Not support for old IR, please use pir mode."
                )

    def _partial_pir_programs(self, program):
        """
        An interface that MUST be implemented by subclasses.
        The return value MUST be two lists, one is a list of types(str), another
        is a list of sub programs.
        For example:
        return [FORWARD, BACKWARD, OPT], [fwd_prog, bwd_prog, opt_prog]
        or
        return [FORWARD], [fwd_prog]
        """
        pass

    def _apply_single_impl(self, main_program, startup_program, context):
        """
        The shared process is implemented in this function and new subclass only need
        to implement two interfaces above, 'create_job_list' and 'partial_programs'.
        """
        raise NotImplementedError("Not support for old IR")

    def _apply_pir_single_impl(self, main_program, startup_program, context):
        """
        The shared process is implemented in this function and new subclass only need
        to implement two interfaces above, 'create_job_list' and 'partial_programs'.
        """

        job_types, sub_programs = self._partial_pir_programs(main_program)

        for i in range(len(job_types)):
            logger.debug(
                f"sub_program type: {job_types[i]}, sub_program:\n{sub_programs[i]}"
            )

        jobs = self._create_job_list()
        type_to_program = set_skip_gc_vars(
            self.get_attr("num_micro_batches"), job_types, sub_programs, jobs
        )

        plan = core.Plan(jobs, type_to_program)
        context.set_attr("plan", plan)

    def _add_dependency(self, recorder_op, waiter_op, name):
        '''
        Add the extra event dependency of the two operators.
        This function mainly aims for the cross-programs in pipeline parallelism,
        especial for the 'send_v2' 'recv_v2' etc.
        '''
        if not recorder_op.has_attr("force_record_event"):
            recorder_op.set_bool_attr("force_record_event", True)
        recorder_op.set_str_attr("event_to_record", name)
        waiter_op.set_str_array_attr("events_to_wait", [name])

    def _add_dependency_if_necessary(
        self,
        type_to_ops,
        cur_job_type,
        next_job_type,
        op_idx,
        rst_idx,
        var_name,
    ):
        if not (
            ("backward" in cur_job_type and "send_backward" in next_job_type)
            or ("recv_forward" in cur_job_type and "forward" in next_job_type)
        ):
            return

        first_used_idx = None
        first_used_op = None
        for used_op in (
            type_to_ops[next_job_type][op_idx].result(rst_idx).all_used_ops()
        ):
            used_idx = type_to_ops[next_job_type].index(used_op)
            if first_used_idx is None or used_idx < first_used_idx:
                first_used_idx = used_idx
                first_used_op = used_op

        if first_used_op is not None:
            self._add_dependency(
                type_to_ops[cur_job_type][op_idx], first_used_op, var_name
            )
