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

"""
Diffusion models deployment module for FastDeploy.

This module provides high-performance inference capabilities for diffusion models
including Stable Diffusion and Flux models with comprehensive optimizations.
"""

from .config import DiffusionConfig
from .predictor import DiffusionPredictor
from .sd_pipeline import SDPipeline
from .flux_pipeline import FluxPipeline
from . import passes
from .tensorrt_integration import DiffusionTensorRTManager, DiffusionTensorRTPlugin

__all__ = [
    'DiffusionConfig',
    'DiffusionPredictor',
    'SDPipeline',
    'FluxPipeline',
    'passes',
    'DiffusionTensorRTManager',
    'DiffusionTensorRTPlugin'
]
