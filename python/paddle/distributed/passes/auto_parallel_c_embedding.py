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

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)


@register_pass("auto_parallel_c_embedding")
class AllreduceMatmulGradOverlappingPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        # 1. check embedding op

        # 2. replace embedding op with c_embedding op

        # 3. update spmd

        pass
