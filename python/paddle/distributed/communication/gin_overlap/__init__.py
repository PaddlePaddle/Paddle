# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Communication primitive of gin_overlap: the reduce_scatter of each sharding-stage1 gradient bucket, done in stages
with NCCL GIN one-sided puts.

This package provides only the primitive: the layout of the windows, the recv pool and the host windows, setting up
and checking the two communication groups, the parameter tables handed to the C++ release layer, and the table of
the 2D-bucket gather to the owner (engine.GinOverlapPrimitive; callers subclass it). When each stage is released is
decided by the caller. Terms and layouts are described in full in the module docstring of engine.py.

The C++ / CUDA implementation lives in paddle/fluid/distributed/collective/gin_overlap/ and is exported as
paddle.base.core.gin_overlap. Builds without it have no such attribute, and device_engine() raises with the build
requirements.

Not a public API: __all__ is empty; take names from here or from .engine.
"""

from .engine import (  # noqa: F401
    GROUP_DENSE,
    GROUP_MOE,
    LOG_EVERY,
    TIMEOUT_S,
    GinOverlapPrimitive,
    check_cuda,
    device_engine,
    log,
)

__all__ = []
