#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Pipeline Parallel Implementation Sub-package.

This sub-package contains the refactored pipeline parallel implementation
split into multiple modules for better maintainability while keeping the
external API unchanged.

Public Classes:
    - FakeMicroDataset: Fake micro-batch dataset for pipeline parallel
    - PipelineDatasetPreprocessor: Wrapper for pipeline dataset
    - NoPipelineParallel: No pipeline parallel implementation
    - PipelineParallel: Base pipeline parallel implementation
    - PipelineParallelWithInterleave: Pipeline parallel with virtual stages
    - PipelineParallelWithInterleaveFthenB: Pipeline parallel with FthenB schedule
    - VPPFhenBInBalancedMemory: Pipeline parallel with memory optimization

Public Enums:
    - PipelineParallelMicroStepLocations: Enum for micro-step locations

Public Functions:
    - register_global_pipeline_parallel_hook: Register global hooks
    - profile_pipeline_details: Profile pipeline memory usage
    - get_action: Get communication action
    - _get_align_mode_scale: Get align mode scale

Public Classes from submodules:
    - P2PAsyncHandle: P2P async communication handle
    - OffloadQueue: Offload queue for memory optimization
    - PipelineParallelMicroStepCallback: Micro-step callback manager
"""

from __future__ import annotations

# Balanced memory pipeline parallel
from .balanced_memory import VPPFhenBInBalancedMemory

# Base pipeline parallel
from .base_pipeline import PipelineParallel

# FthenB pipeline parallel
from .fthenb_pipeline import PipelineParallelWithInterleaveFthenB

# Hook system
from .hook_system import (
    PipelineParallelMicroStepCallback,
    PipelineParallelMicroStepLocations,
    pipeline_parallel_callbacks_,
    register_global_pipeline_parallel_hook,
)

# Interleave pipeline parallel
from .interleave_pipeline import PipelineParallelWithInterleave

# No pipeline parallel
from .no_pipeline import NoPipelineParallel

# P2P communication
from .p2p_handle import OffloadQueue, P2PAsyncHandle

# Utility functions and classes
from .utils import (
    FakeMicroDataset,
    PipelineDatasetPreprocessor,
    _get_align_mode_scale,
    get_action,
    profile_pipeline_details,
)

__all__ = [
    # Utility functions and classes
    "FakeMicroDataset",
    "PipelineDatasetPreprocessor",
    "profile_pipeline_details",
    "get_action",
    "_get_align_mode_scale",
    # Hook system
    "PipelineParallelMicroStepLocations",
    "PipelineParallelMicroStepCallback",
    "register_global_pipeline_parallel_hook",
    "pipeline_parallel_callbacks_",
    # P2P communication
    "P2PAsyncHandle",
    "OffloadQueue",
    # Pipeline parallel classes
    "NoPipelineParallel",
    "PipelineParallel",
    "PipelineParallelWithInterleave",
    "PipelineParallelWithInterleaveFthenB",
    "VPPFhenBInBalancedMemory",
]
