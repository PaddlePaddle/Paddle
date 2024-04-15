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

import os

from ..pass_base import PassContext, new_pass
from .pipeline_1f1b import Pipeline1F1BPass  # noqa: F401
from .pipeline_eager_1f1b import PipelineEager1F1BPass  # noqa: F401
from .pipeline_fthenb import PipelineFThenBPass  # noqa: F401
from .pipeline_vpp import PipelineVirtualPipelinePass  # noqa: F401
from .pipeline_zero_bubble import PipelineZeroBubblePipelinePass  # noqa: F401

__all__ = []


def apply_pass(main_program, startup_program, pass_name, pass_attr={}):
    assert pass_name in [
        "FThenB",
        "1F1B",
        "Eager1F1B",
        "VPP",
        "ZBH1",
    ], f"pipeline scheduler only support FThenB, 1F1B, Eager1F1B, VPP and ZBH1, but receive {pass_name}"

    if pass_name == "1F1B":
        # TODO(Ruibiao): Move FLAGS_1f1b_backward_forward_overlap and
        # FLAGS_mp_async_allreduce_in_backward to auto parallel Strategy
        # after these two optimizations are available.
        pass_attr["enable_backward_forward_overlap"] = int(
            os.environ.get("FLAGS_1f1b_backward_forward_overlap", 0)
        )

    pipeline_pass = new_pass("pipeline_scheduler_" + pass_name, pass_attr)
    pass_context = PassContext()
    pipeline_pass.apply([main_program], [startup_program], pass_context)
    plan = pass_context.get_attr("plan")
    return plan
