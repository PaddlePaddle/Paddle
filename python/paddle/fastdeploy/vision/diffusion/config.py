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
Configuration classes for diffusion models in FastDeploy.
"""

import os
from typing import Dict, List, Optional, Union, Any
import warnings


class DiffusionConfig:
    """
    Configuration class for diffusion model deployment.

    This class encapsulates all configuration parameters needed for
    deploying diffusion models including Stable Diffusion and Flux.

    Args:
        model_path (str): Path to the diffusion model directory
        model_type (str): Type of diffusion model ('stable-diffusion' or 'flux')
        device (str): Target device ('cpu', 'gpu', 'xpu', etc.)
        use_fp16 (bool): Whether to use FP16 precision for inference
        use_tensorrt (bool): Whether to use TensorRT acceleration
        use_cinn (bool): Whether to use CINN optimization
        max_batch_size (int): Maximum batch size for inference
        height (int): Default height for generated images
        width (int): Default width for generated images
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Guidance scale for classifier-free guidance
        enable_memory_optimization (bool): Whether to enable memory optimization
        enable_dynamic_shape (bool): Whether to enable dynamic shape support
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "stable-diffusion",
        device: str = "gpu",
        use_fp16: bool = True,
        use_tensorrt: bool = False,
        use_cinn: bool = True,
        max_batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        enable_memory_optimization: bool = True,
        enable_dynamic_shape: bool = True,
        **kwargs
    ):
        # 基础配置
        self.model_path = model_path
        self.model_type = model_type
        self.device = device

        # 精度和优化配置
        self.use_fp16 = use_fp16
        self.use_tensorrt = use_tensorrt
        self.use_cinn = use_cinn

        # 推理配置
        self.max_batch_size = max_batch_size
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # 高级配置
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_dynamic_shape = enable_dynamic_shape

        # 额外的配置参数
        self._extra_config = kwargs

        # 验证配置
        self._validate_config()

    def _validate_config(self):
        """验证配置参数的有效性"""
        # 验证模型路径
        if not os.path.exists(self.model_path):
            warnings.warn(f"Model path {self.model_path} does not exist")

        # 验证模型类型
        supported_model_types = ["stable-diffusion", "flux", "sdxl"]
        if self.model_type not in supported_model_types:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Supported types: {supported_model_types}"
            )

        # 验证设备
        supported_devices = ["cpu", "gpu", "xpu"]
        if self.device not in supported_devices:
            raise ValueError(
                f"Unsupported device: {self.device}. "
                f"Supported devices: {supported_devices}"
            )

        # 验证推理步骤
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")

        # 验证图像尺寸
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")

        # TensorRT只在GPU上支持
        if self.use_tensorrt and self.device != "gpu":
            raise ValueError("TensorRT is only supported on GPU")

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        config_dict = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "use_tensorrt": self.use_tensorrt,
            "use_cinn": self.use_cinn,
            "max_batch_size": self.max_batch_size,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_memory_optimization": self.enable_memory_optimization,
            "enable_dynamic_shape": self.enable_dynamic_shape,
        }
        config_dict.update(self._extra_config)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DiffusionConfig":
        """从字典创建配置对象"""
        return cls(**config_dict)

    def __repr__(self) -> str:
        return f"DiffusionConfig(model_type='{self.model_type}', device='{self.device}')"
