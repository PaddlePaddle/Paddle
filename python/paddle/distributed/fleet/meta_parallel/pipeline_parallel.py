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
"""Pipeline Parallel Implementation.

This module provides pipeline parallelism for distributed training.
The implementation has been refactored into sub-modules for better maintainability.

All public interfaces are re-exported from the pipeline_impl sub-package,
ensuring backward compatibility with existing code.
"""

from __future__ import annotations

# Re-export all public interfaces from pipeline_impl sub-package
from .pipeline_impl import (
    # Utility functions and classes
    FakeMicroDataset,
    # Pipeline parallel classes
    NoPipelineParallel,
    OffloadQueue,
    # P2P communication
    P2PAsyncHandle,
    PipelineDatasetPreprocessor,
    PipelineParallel,
    PipelineParallelMicroStepCallback,
    # Hook system
    PipelineParallelMicroStepLocations,
    PipelineParallelWithInterleave,
    PipelineParallelWithInterleaveFthenB,
    VPPFhenBInBalancedMemory,
    _get_align_mode_scale,
    get_action,
    pipeline_parallel_callbacks_,
    profile_pipeline_details,
    register_global_pipeline_parallel_hook,
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
