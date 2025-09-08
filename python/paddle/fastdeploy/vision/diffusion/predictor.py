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
Base predictor class for diffusion models in FastDeploy.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod

import paddle
from paddle.inference import Config, create_predictor, PaddlePredictor

from .config import DiffusionConfig


class DiffusionPredictor(PaddlePredictor, ABC):
    """
    Base predictor class for diffusion models.

    This class inherits from PaddlePaddle's AnalysisPredictor and provides
    specialized optimizations for diffusion models including multi-stage pipeline
    support, memory optimization, and dynamic shape handling.

    Supports multi-stage inference pipeline: text encoding -> denoising -> decoding

    Args:
        config (DiffusionConfig): Configuration object for the predictor
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config

        # 性能监控
        self.inference_times = []
        self.memory_usage = []

        # 多阶段pipeline组件
        self.text_encoder = None
        self.denoising_model = None
        self.decoder = None

        # 初始化预测器
        self._initialize_predictor()

    def _initialize_predictor(self):
        """初始化PaddlePaddle预测器"""
        try:
            # 创建推理配置
            inference_config = Config()

            # 设置模型路径和参数
            self._setup_model_paths(inference_config)

            # 设置设备和硬件配置
            self._setup_device_config(inference_config)

            # 设置精度和优化配置
            self._setup_precision_config(inference_config)

            # 设置内存和性能优化
            self._setup_optimization_config(inference_config)

            # 设置TensorRT配置（如果启用）
            if self.config.use_tensorrt:
                self._setup_tensorrt_config(inference_config)

            # 调用父类初始化
            super().__init__(inference_config)

            # 初始化多阶段pipeline
            self._setup_pipeline()

            print(f"✅ DiffusionPredictor initialized successfully for {self.config.model_type}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize DiffusionPredictor: {e}")

    def _setup_model_paths(self, inference_config: Config):
        """设置模型文件路径"""
        if os.path.isfile(self.config.model_path):
            # 单个模型文件
            inference_config.set_model(self.config.model_path)
        elif os.path.isdir(self.config.model_path):
            # 模型目录
            model_file = os.path.join(self.config.model_path, "__model__")
            params_file = os.path.join(self.config.model_path, "__params__")

            if os.path.exists(model_file) and os.path.exists(params_file):
                inference_config.set_model(model_file, params_file)
            else:
                # 尝试其他常见格式
                self._try_alternative_model_formats(inference_config)
        else:
            raise FileNotFoundError(f"Model path not found: {self.config.model_path}")

    def _try_alternative_model_formats(self, inference_config: Config):
        """尝试其他模型文件格式"""
        model_dir = self.config.model_path

        # 尝试Paddle静态图格式
        if os.path.exists(os.path.join(model_dir, "model.pdmodel")):
            model_file = os.path.join(model_dir, "model.pdmodel")
            params_file = os.path.join(model_dir, "model.pdiparams")
            inference_config.set_model(model_file, params_file)
            return

        # 尝试ONNX格式（如果有转换器）
        onnx_file = os.path.join(model_dir, "model.onnx")
        if os.path.exists(onnx_file):
            inference_config.set_model(onnx_file)
            return

        raise FileNotFoundError(f"No valid model file found in {model_dir}")

    def _setup_device_config(self, inference_config: Config):
        """设置设备配置"""
        if self.config.device == "gpu":
            # GPU配置
            gpu_id = 0  # 可以从配置中获取
            gpu_mem_pool_init_size_mb = 1000  # 1GB初始内存池
            gpu_mem_pool_max_size_mb = 2000  # 2GB最大内存池

            inference_config.enable_use_gpu(gpu_mem_pool_init_size_mb, gpu_id)
            inference_config.set_gpu_device_id(gpu_id)

            # 设置GPU内存配置
            if hasattr(inference_config, 'set_gpu_mem_pool_config'):
                inference_config.set_gpu_mem_pool_config(
                    gpu_mem_pool_init_size_mb, gpu_mem_pool_max_size_mb
                )

        elif self.config.device == "xpu":
            # XPU配置
            inference_config.enable_xpu()
            inference_config.set_xpu_device_id(0)
        else:
            # CPU配置
            inference_config.disable_gpu()
            inference_config.set_cpu_math_library_num_threads(8)  # 使用8个线程

            # 启用MKLDNN
            inference_config.enable_mkldnn()

    def _setup_precision_config(self, inference_config: Config):
        """设置精度配置"""
        if self.config.use_fp16:
            # 启用半精度推理
            inference_config.enable_mkldnn()
            inference_config.enable_mkldnn_bfloat16()

            # 设置TensorRT半精度（如果使用TensorRT）
            if self.config.use_tensorrt:
                inference_config.enable_tensorrt_half_precision()

        elif self.config.use_tensorrt:
            # TensorRT默认使用FP32
            inference_config.disable_tensorrt_half_precision()

    def _setup_optimization_config(self, inference_config: Config):
        """设置优化配置"""
        # 启用内存优化
        if self.config.enable_memory_optimization:
            inference_config.enable_memory_optim()
            inference_config.enable_ir_optim()

            # 设置内存优化级别
            inference_config.set_optim_cache_dir("/tmp/paddle_optim_cache")

        # 启用计算图优化
        inference_config.enable_ir_optim()
        inference_config.set_ir_optim(True)

        # 设置优化Pass
        optimization_passes = [
            "adaptive_pool2d_convert_global_pass",  # 全局池化优化
            "shuffle_channel_detect_pass",         # 通道shuffle优化
            "quant_conv2d_dequant_fuse_pass",      # 量化卷积融合
            "conv_bn_fuse_pass",                   # 卷积批归一化融合
            "conv_elementwise_add_fuse_pass",      # 卷积元素加法融合
        ]

        for pass_name in optimization_passes:
            try:
                inference_config.pass_builder().append_pass(pass_name)
            except:
                # 某些pass可能不存在，跳过
                continue

    def _setup_tensorrt_config(self, inference_config: Config):
        """设置TensorRT配置"""
        if not self.config.use_tensorrt:
            return

        # 启用TensorRT
        inference_config.enable_tensorrt_engine(
            workspace_size=1 << 30,  # 1GB workspace
            max_batch_size=self.config.max_batch_size,
            min_subgraph_size=3,     # 最小子图大小
            precision_mode=self._get_tensorrt_precision_mode(),
            use_static=True,         # 使用静态shape
            use_calib_mode=False     # 不使用量化校准
        )

        # 设置TensorRT缓存目录
        inference_config.set_trt_cache_dir("/tmp/paddle_trt_cache")

        # 启用动态shape（如果配置了）
        if self.config.enable_dynamic_shape:
            inference_config.enable_tensorrt_dla()

    def _get_tensorrt_precision_mode(self) -> str:
        """获取TensorRT精度模式"""
        if self.config.use_fp16:
            return "FP16"
        else:
            return "FP32"

    def _setup_pipeline(self):
        """设置多阶段推理pipeline"""
        try:
            # 获取输入输出名称
            input_names = self.get_input_names()
            output_names = self.get_output_names()

            print(f"Pipeline inputs: {input_names}")
            print(f"Pipeline outputs: {output_names}")

            # 验证pipeline配置
            self._validate_pipeline_config()

            # 初始化pipeline组件
            self._initialize_pipeline_components()

            # 设置pipeline执行顺序
            self._setup_pipeline_execution_order()

            # 验证pipeline完整性
            self._validate_pipeline_integrity()

            print("✅ Pipeline setup completed successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to setup pipeline: {e}")

    def _validate_pipeline_config(self):
        """验证pipeline配置"""
        required_inputs = []
        required_outputs = []

        # 根据模型类型设置必需的输入输出
        if self.config.model_type == "stable-diffusion":
            required_inputs = ["input_ids", "latent_sample", "timestep"]
            required_outputs = ["sample"]
        elif self.config.model_type == "flux":
            required_inputs = ["input_ids", "latent_sample", "timestep", "guidance"]
            required_outputs = ["sample"]

        # 检查必需的输入
        input_names = self.get_input_names()
        for required_input in required_inputs:
            if required_input not in input_names:
                print(f"⚠️  Warning: Required input '{required_input}' not found in model")

        # 检查必需的输出
        output_names = self.get_output_names()
        for required_output in required_outputs:
            if required_output not in output_names:
                print(f"⚠️  Warning: Required output '{required_output}' not found in model")

    def _initialize_pipeline_components(self):
        """初始化pipeline组件"""
        # 根据模型类型初始化相应的组件
        if self.config.model_type == "stable-diffusion":
            self._initialize_sd_components()
        elif self.config.model_type == "flux":
            self._initialize_flux_components()
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def _initialize_sd_components(self):
        """初始化Stable Diffusion组件"""
        # Stable Diffusion通常有以下组件：
        # 1. CLIP文本编码器
        # 2. U-Net去噪模型
        # 3. VAE解码器

        # 检查组件是否存在
        model_dir = self.config.model_path

        # 检查文本编码器
        text_encoder_path = os.path.join(model_dir, "text_encoder")
        if os.path.exists(text_encoder_path):
            self.text_encoder = self._create_sub_predictor(text_encoder_path)
            print("✅ Text encoder initialized")

        # 检查U-Net
        unet_path = os.path.join(model_dir, "unet")
        if os.path.exists(unet_path):
            self.denoising_model = self._create_sub_predictor(unet_path)
            print("✅ U-Net denoising model initialized")

        # 检查VAE解码器
        vae_path = os.path.join(model_dir, "vae_decoder")
        if os.path.exists(vae_path):
            self.decoder = self._create_sub_predictor(vae_path)
            print("✅ VAE decoder initialized")

    def _initialize_flux_components(self):
        """初始化Flux组件"""
        # Flux通常有以下组件：
        # 1. T5文本编码器
        # 2. DiT Transformer去噪模型
        # 3. VAE解码器

        model_dir = self.config.model_path

        # 检查T5文本编码器
        t5_path = os.path.join(model_dir, "text_encoder")
        if os.path.exists(t5_path):
            self.text_encoder = self._create_sub_predictor(t5_path)
            print("✅ T5 text encoder initialized")

        # 检查DiT Transformer
        transformer_path = os.path.join(model_dir, "transformer")
        if os.path.exists(transformer_path):
            self.denoising_model = self._create_sub_predictor(transformer_path)
            print("✅ DiT transformer initialized")

        # 检查VAE解码器
        vae_path = os.path.join(model_dir, "vae_decoder")
        if os.path.exists(vae_path):
            self.decoder = self._create_sub_predictor(vae_path)
            print("✅ VAE decoder initialized")

    def _create_sub_predictor(self, model_path: str) -> 'DiffusionPredictor':
        """创建子预测器"""
        sub_config = DiffusionConfig(
            model_path=model_path,
            model_type=self.config.model_type,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            use_tensorrt=self.config.use_tensorrt,
            use_cinn=self.config.use_cinn,
            max_batch_size=self.config.max_batch_size,
            enable_memory_optimization=self.config.enable_memory_optimization,
            enable_dynamic_shape=self.config.enable_dynamic_shape,
        )

        # 创建子预测器（注意：这里可能需要递归调用，需要小心处理）
        # 为了避免递归，我们使用基础的PaddlePredictor
        from paddle.inference import create_predictor

        predictor_config = Config()
        if os.path.exists(os.path.join(model_path, "__model__")):
            model_file = os.path.join(model_path, "__model__")
            params_file = os.path.join(model_path, "__params__")
            predictor_config.set_model(model_file, params_file)
        elif os.path.exists(os.path.join(model_path, "model.pdmodel")):
            model_file = os.path.join(model_path, "model.pdmodel")
            params_file = os.path.join(model_path, "model.pdiparams")
            predictor_config.set_model(model_file, params_file)

        # 应用相同的设备和精度设置
        if sub_config.device == "gpu":
            predictor_config.enable_use_gpu(100, 0)
        elif sub_config.device == "xpu":
            predictor_config.enable_xpu()
        else:
            predictor_config.disable_gpu()

        if sub_config.use_fp16:
            predictor_config.enable_mkldnn_bfloat16()

        return create_predictor(predictor_config)

    def _setup_pipeline_execution_order(self):
        """设置pipeline执行顺序"""
        # 定义pipeline阶段的执行顺序
        self.pipeline_stages = [
            "text_encoding",
            "denoising",
            "decoding"
        ]

        # 设置阶段间的依赖关系
        self.stage_dependencies = {
            "denoising": ["text_encoding"],
            "decoding": ["denoising"]
        }

        print("✅ Pipeline execution order configured")

    def _validate_pipeline_integrity(self):
        """验证pipeline完整性"""
        # 检查必需的组件是否存在
        missing_components = []

        if self.text_encoder is None:
            missing_components.append("text_encoder")

        if self.denoising_model is None:
            missing_components.append("denoising_model")

        if self.decoder is None:
            missing_components.append("decoder")

        if missing_components:
            print(f"⚠️  Warning: Missing components: {missing_components}")
            print("Pipeline will use fallback implementations where possible")
        else:
            print("✅ All pipeline components available")

    @abstractmethod
    def encode_text(self, text_inputs: Dict[str, Any]) -> paddle.Tensor:
        """
        第一阶段：文本编码

        Args:
            text_inputs: 文本输入数据

        Returns:
            文本embeddings
        """
        pass

    @abstractmethod
    def denoise(self, latents: paddle.Tensor, text_embeddings: paddle.Tensor,
                num_inference_steps: int, guidance_scale: float) -> paddle.Tensor:
        """
        第二阶段：去噪过程

        Args:
            latents: 初始噪声latents
            text_embeddings: 文本embeddings
            num_inference_steps: 推理步数
            guidance_scale: 引导尺度

        Returns:
            去噪后的latents
        """
        pass

    @abstractmethod
    def decode_image(self, latents: paddle.Tensor) -> np.ndarray:
        """
        第三阶段：图像解码

        Args:
            latents: 去噪后的latents

        Returns:
            解码后的图像
        """
        pass

    def run_pipeline(self, inputs: Dict[str, Any]) -> Any:
        """
        执行完整的三阶段推理pipeline

        Args:
            inputs: 输入数据

        Returns:
            推理结果
        """
        # 记录开始时间
        start_time = time.time()

        # 第一阶段：文本编码
        text_embeddings = self.encode_text(inputs)

        # 第二阶段：去噪
        num_steps = inputs.get('num_inference_steps', self.config.num_inference_steps)
        guidance_scale = inputs.get('guidance_scale', self.config.guidance_scale)

        # 生成初始latents
        latents = self._prepare_latents(inputs)

        # 执行去噪
        denoised_latents = self.denoise(latents, text_embeddings, num_steps, guidance_scale)

        # 第三阶段：图像解码
        result = self.decode_image(denoised_latents)

        # 记录性能指标
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        return result

    def _prepare_latents(self, inputs: Dict[str, Any]) -> paddle.Tensor:
        """准备初始latents"""
        height = inputs.get('height', self.config.height)
        width = inputs.get('width', self.config.width)

        # 计算latent空间的尺寸
        latent_height = height // 8
        latent_width = width // 8

        # 生成随机噪声（子类可以重写这个方法）
        latents = paddle.randn([1, 4, latent_height, latent_width])
        return latents

    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if not self.inference_times:
            return {"avg_inference_time": 0.0, "total_inferences": 0}

        return {
            "avg_inference_time": np.mean(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "total_inferences": len(self.inference_times),
        }

    def clear_performance_stats(self):
        """清除性能统计信息"""
        self.inference_times.clear()
        self.memory_usage.clear()
