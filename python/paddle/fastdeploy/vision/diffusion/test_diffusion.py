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
Unit tests for FastDeploy Diffusion Models.
"""

import unittest
import tempfile
import os
import numpy as np
from PIL import Image

import paddle

from .config import DiffusionConfig
from .sd_pipeline import SDPipeline, DDPMScheduler
from .flux_pipeline import FluxPipeline, FlowScheduler


class TestDiffusionConfig(unittest.TestCase):
    """测试DiffusionConfig类"""

    def test_config_creation(self):
        """测试配置创建"""
        config = DiffusionConfig(
            model_path="/tmp/test_model",
            model_type="stable-diffusion",
            device="gpu",
        )
        self.assertEqual(config.model_path, "/tmp/test_model")
        self.assertEqual(config.model_type, "stable-diffusion")
        self.assertEqual(config.device, "gpu")

    def test_config_validation(self):
        """测试配置验证"""
        # 测试无效的模型类型
        with self.assertRaises(ValueError):
            DiffusionConfig(
                model_path="/tmp/test_model",
                model_type="invalid-model",
            )

        # 测试无效的设备
        with self.assertRaises(ValueError):
            DiffusionConfig(
                model_path="/tmp/test_model",
                model_type="stable-diffusion",
                device="invalid-device",
            )

    def test_config_to_dict(self):
        """测试配置序列化"""
        config = DiffusionConfig(
            model_path="/tmp/test_model",
            model_type="stable-diffusion",
        )
        config_dict = config.to_dict()
        self.assertIn("model_path", config_dict)
        self.assertIn("model_type", config_dict)

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        config_dict = {
            "model_path": "/tmp/test_model",
            "model_type": "stable-diffusion",
            "device": "cpu",
        }
        config = DiffusionConfig.from_dict(config_dict)
        self.assertEqual(config.model_path, "/tmp/test_model")


class TestDDPMScheduler(unittest.TestCase):
    """测试DDPM调度器"""

    def test_scheduler_creation(self):
        """测试调度器创建"""
        scheduler = DDPMScheduler()
        self.assertEqual(scheduler.num_train_timesteps, 1000)
        self.assertIsNotNone(scheduler.betas)
        self.assertIsNotNone(scheduler.alphas)

    def test_set_timesteps(self):
        """测试设置时间步"""
        scheduler = DDPMScheduler()
        scheduler.set_timesteps(10)
        self.assertEqual(len(scheduler.timesteps), 10)

    def test_scheduler_step(self):
        """测试调度器步骤"""
        scheduler = DDPMScheduler()
        scheduler.set_timesteps(10)

        # 创建测试输入
        model_output = paddle.randn([1, 4, 64, 64])
        timestep = 5
        sample = paddle.randn([1, 4, 64, 64])

        result = scheduler.step(model_output, timestep, sample)
        self.assertEqual(result.shape, sample.shape)


class TestFlowScheduler(unittest.TestCase):
    """测试Flow调度器"""

    def test_scheduler_creation(self):
        """测试调度器创建"""
        scheduler = FlowScheduler()
        self.assertEqual(scheduler.num_train_timesteps, 1000)

    def test_set_timesteps(self):
        """测试设置时间步"""
        scheduler = FlowScheduler()
        scheduler.set_timesteps(10)
        self.assertEqual(len(scheduler.timesteps), 10)

    def test_scheduler_step(self):
        """测试调度器步骤"""
        scheduler = FlowScheduler()
        scheduler.set_timesteps(10)

        # 创建测试输入
        model_output = paddle.randn([1, 16, 64, 64])
        timestep = 0.5
        sample = paddle.randn([1, 16, 64, 64])

        result = scheduler.step(model_output, timestep, sample)
        self.assertEqual(result.shape, sample.shape)


class TestSDPipeline(unittest.TestCase):
    """测试Stable Diffusion Pipeline"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

        # 创建模拟的tokenizer配置
        tokenizer_dir = os.path.join(self.temp_dir, "tokenizer")
        os.makedirs(tokenizer_dir)
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            f.write('{"max_position_embeddings": 77, "vocab_size": 49408}')

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_pipeline_creation(self):
        """测试pipeline创建"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="stable-diffusion",
        )
        pipeline = SDPipeline(config)
        self.assertIsNotNone(pipeline.scheduler)
        self.assertIsNotNone(pipeline.tokenizer_config)

    def test_encode_prompt(self):
        """测试文本编码"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="stable-diffusion",
        )
        pipeline = SDPipeline(config)

        prompt = "test prompt"
        embeddings = pipeline._encode_prompt(prompt)
        self.assertEqual(len(embeddings.shape), 3)  # [batch, seq, hidden]

    def test_prepare_latents(self):
        """测试latent准备"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="stable-diffusion",
        )
        pipeline = SDPipeline(config)

        latents = pipeline._prepare_latents(512, 512)
        self.assertEqual(latents.shape, [1, 4, 64, 64])  # SD的latent维度


class TestFluxPipeline(unittest.TestCase):
    """测试Flux Pipeline"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

        # 创建模拟的tokenizer配置
        tokenizer_dir = os.path.join(self.temp_dir, "tokenizer")
        os.makedirs(tokenizer_dir)
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            f.write('{"max_position_embeddings": 256, "vocab_size": 49408}')

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_pipeline_creation(self):
        """测试pipeline创建"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="flux",
        )
        pipeline = FluxPipeline(config)
        self.assertIsNotNone(pipeline.scheduler)
        self.assertIsNotNone(pipeline.tokenizer_config)

    def test_encode_prompt(self):
        """测试文本编码"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="flux",
        )
        pipeline = FluxPipeline(config)

        prompt = "test prompt"
        embeddings = pipeline._encode_prompt(prompt)
        self.assertEqual(len(embeddings.shape), 3)  # [batch, seq, hidden]

    def test_prepare_latents(self):
        """测试latent准备"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="flux",
        )
        pipeline = FluxPipeline(config)

        latents = pipeline._prepare_latents(1024, 1024)
        self.assertEqual(latents.shape, [1, 16, 64, 64])  # Flux的latent维度

    def test_get_timestep_embedding(self):
        """测试时间步嵌入"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="flux",
        )
        pipeline = FluxPipeline(config)

        timestep = paddle.to_tensor([0.5])
        embedding = pipeline._get_timestep_embedding(timestep)
        self.assertEqual(len(embedding.shape), 2)  # [batch, hidden]


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

        # 创建模拟的tokenizer配置
        tokenizer_dir = os.path.join(self.temp_dir, "tokenizer")
        os.makedirs(tokenizer_dir)
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            f.write('{"max_position_embeddings": 77, "vocab_size": 49408}')

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline_sd(self):
        """测试完整的SD pipeline"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="stable-diffusion",
            height=256,  # 使用小尺寸进行测试
            width=256,
            num_inference_steps=1,  # 只测试一个步骤
        )
        pipeline = SDPipeline(config)

        # 生成图像（会使用模拟的推理）
        image = pipeline.text_to_image("test prompt", seed=42)

        # 验证输出
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (256, 256))

    def test_full_pipeline_flux(self):
        """测试完整的Flux pipeline"""
        config = DiffusionConfig(
            model_path=self.temp_dir,
            model_type="flux",
            height=256,
            width=256,
            num_inference_steps=1,
        )
        pipeline = FluxPipeline(config)

        # 生成图像
        image = pipeline.text_to_image("test prompt", seed=42)

        # 验证输出
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (256, 256))


if __name__ == '__main__':
    unittest.main()
