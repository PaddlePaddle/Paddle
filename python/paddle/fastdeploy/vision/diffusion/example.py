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
Example usage of FastDeploy Diffusion Models.
"""

import os
from PIL import Image
import numpy as np

from .config import DiffusionConfig
from .sd_pipeline import SDPipeline
from .flux_pipeline import FluxPipeline


def stable_diffusion_example():
    """Stable Diffusion使用示例"""
    print("=== Stable Diffusion Example ===")

    # 创建配置
    config = DiffusionConfig(
        model_path="/path/to/stable-diffusion-model",
        model_type="stable-diffusion",
        device="gpu",
        use_fp16=True,
        use_cinn=True,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=7.5,
    )

    # 创建pipeline
    pipeline = SDPipeline(config)

    # 生成图像
    prompt = "A beautiful landscape with mountains and lake"
    negative_prompt = "blurry, low quality"

    image = pipeline.text_to_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=42,
    )

    # 保存图像
    image.save("stable_diffusion_output.png")
    print("Generated image saved as stable_diffusion_output.png")


def flux_example():
    """Flux模型使用示例"""
    print("=== Flux Example ===")

    # 创建配置
    config = DiffusionConfig(
        model_path="/path/to/flux-model",
        model_type="flux",
        device="gpu",
        use_fp16=True,
        use_cinn=True,
        height=1024,
        width=1024,
        num_inference_steps=28,  # Flux通常使用更多步骤
        guidance_scale=3.5,  # Flux使用较低的guidance
    )

    # 创建pipeline
    pipeline = FluxPipeline(config)

    # 生成图像
    prompt = "A futuristic city at sunset, highly detailed"

    image = pipeline.text_to_image(
        prompt=prompt,
        seed=123,
    )

    # 保存图像
    image.save("flux_output.png")
    print("Generated image saved as flux_output.png")


def performance_comparison():
    """性能对比示例"""
    print("=== Performance Comparison ===")

    # Stable Diffusion配置
    sd_config = DiffusionConfig(
        model_path="/path/to/stable-diffusion-model",
        model_type="stable-diffusion",
        device="gpu",
        use_fp16=True,
        use_cinn=True,
    )

    # Flux配置
    flux_config = DiffusionConfig(
        model_path="/path/to/flux-model",
        model_type="flux",
        device="gpu",
        use_fp16=True,
        use_cinn=True,
    )

    # 创建pipelines
    sd_pipeline = SDPipeline(sd_config)
    flux_pipeline = FluxPipeline(flux_config)

    prompt = "A cat wearing sunglasses"

    # 测试Stable Diffusion
    import time
    start_time = time.time()
    sd_image = sd_pipeline.text_to_image(prompt, num_inference_steps=20)
    sd_time = time.time() - start_time

    # 测试Flux
    start_time = time.time()
    flux_image = flux_pipeline.text_to_image(prompt, num_inference_steps=28)
    flux_time = time.time() - start_time

    print(".2f")
    print(".2f")
    print(".2f")


def custom_model_example():
    """自定义模型配置示例"""
    print("=== Custom Model Configuration ===")

    # 从字典创建配置
    config_dict = {
        "model_path": "/path/to/custom-model",
        "model_type": "stable-diffusion",
        "device": "gpu",
        "use_fp16": True,
        "use_tensorrt": True,
        "max_batch_size": 4,
        "height": 768,
        "width": 768,
        "num_inference_steps": 25,
        "guidance_scale": 8.0,
        "enable_memory_optimization": True,
        "enable_dynamic_shape": True,
    }

    config = DiffusionConfig.from_dict(config_dict)
    print(f"Custom config: {config}")

    # 创建pipeline
    pipeline = SDPipeline(config)

    # 批量生成
    prompts = [
        "A sunset over the ocean",
        "A forest in autumn",
        "A mountain landscape",
        "A city skyline at night"
    ]

    for i, prompt in enumerate(prompts):
        image = pipeline.text_to_image(prompt, seed=i)
        image.save(f"custom_output_{i}.png")
        print(f"Generated image {i+1}/4")


if __name__ == "__main__":
    print("FastDeploy Diffusion Models Examples")
    print("===================================")

    # 运行示例（需要有效的模型路径）
    try:
        # stable_diffusion_example()
        # flux_example()
        # performance_comparison()
        custom_model_example()
    except Exception as e:
        print(f"Example failed: {e}")
        print("Please ensure model paths are correctly configured.")
