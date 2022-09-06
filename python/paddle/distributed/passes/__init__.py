# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .pass_base import new_pass, PassManager, PassContext
from .fuse_all_reduce import *
from .auto_parallel_gradient_merge import *
from .auto_parallel_sharding import *
from .auto_parallel_amp import *
from .auto_parallel_fp16 import *
from .auto_parallel_recompute import *
from .auto_parallel_quantization import *
from .auto_parallel_data_parallel_optimization import *
from .auto_parallel_grad_clip import *
from .cpp_pass import *
import os
from .ps_trainer_pass import *
from .ps_server_pass import *

__all__ = [
    'new_pass',
    'PassManager',
    'PassContext',
]
