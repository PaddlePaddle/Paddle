# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from .dynamic_dims_inference import DynamicDimensionsInference
from .completion import Completer
from .dist_context import get_default_distributed_context
from .utils import print_program_with_dist_attr
from .tuner.parallel_tuner import ParallelTuner

# from .tuner.parallel_tuner import ParallelTuner


class Planner:

    def __init__(self, mode, dist_context):
        self._mode = mode
        self._dist_context = dist_context

        # NOTE: [HighOrderGrad]. There are grad ops in forward phase, and it need
        # dependency of backward-forward ops in forward completion.
        # TODO: The id mapping will be lost if we clone the original program.
        default_ctx = get_default_distributed_context()
        self._dist_context._dist_op_context = default_ctx.dist_op_context
        self._dist_context.initialize()

        self._dynamic_dims_inference = DynamicDimensionsInference(
            self._dist_context)

        self._completer = Completer(self._dist_context)

        self._strategy = dist_context.strategy
        if self._strategy.auto_search:
            self._parallel_tuner = ParallelTuner(self._dist_context,
                                                 mode=self._mode)

    @property
    def completer(self):
        return self._completer

    def plan(self):
        # Find the dynamic dims
        self._dynamic_dims_inference.infer_dynamic_dims()

        # Complete the dist attr
        if self._strategy.auto_search:
            self._parallel_tuner.tune()
        else:
            self._completer.complete_forward_annotation()

        # parse forward sub block
        self._dist_context.block_state.parse_forward_blocks(
            self._dist_context.serial_main_program)
