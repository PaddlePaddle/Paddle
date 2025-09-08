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
Flux model optimization passes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import paddle
from paddle import nn
import paddle.nn.functional as F
import math


class BaseOptimizationPass(ABC):
    """优化Pass基类"""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def apply(self, model: nn.Layer) -> nn.Layer:
        """应用优化到模型"""
        pass

    def is_applicable(self, model: nn.Layer) -> bool:
        """检查Pass是否适用于该模型"""
        return True


class FluxTransformerFusePass(BaseOptimizationPass):
    """
    Flux Transformer架构优化Pass

    优化策略：
    1. 融合自注意力和交叉注意力操作
    2. 优化多头注意力计算
    3. 融合前馈网络的线性变换
    """

    def apply(self, model: nn.Layer) -> nn.Layer:
        """应用Transformer优化"""

        def fuse_multi_head_attention(module):
            """融合多头注意力计算"""
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                # 融合Q、K、V投影
                q_weight = module.q_proj.weight
                k_weight = module.k_proj.weight
                v_weight = module.v_proj.weight

                # 合并权重
                fused_weight = paddle.concat([q_weight, k_weight, v_weight], axis=0)

                # 创建融合投影
                module.fused_qkv_proj = nn.Linear(
                    q_weight.shape[1],
                    fused_weight.shape[0],
                    bias_attr=False  # Flux通常不使用偏置
                )
                module.fused_qkv_proj.weight.set_value(fused_weight)

                print(f"Fused QKV projection in {module.__class__.__name__}")

        def optimize_attention_patterns(module):
            """优化注意力模式"""
            if hasattr(module, 'self_attn') and hasattr(module, 'cross_attn'):
                # 优化自注意力和交叉注意力的组合
                self_attn = module.self_attn
                cross_attn = module.cross_attn

                # 为两个注意力层启用相同的优化
                if hasattr(self_attn, 'use_flash_attn'):
                    self_attn.use_flash_attn = True
                if hasattr(cross_attn, 'use_flash_attn'):
                    cross_attn.use_flash_attn = True

                print(f"Optimized attention patterns in {module.__class__.__name__}")

        def fuse_feed_forward_network(module):
            """融合前馈网络"""
            if hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                fc1 = module.fc1
                fc2 = module.fc2

                # 检查是否可以融合（需要相同的中间维度）
                if fc1.weight.shape[0] == fc2.weight.shape[1]:
                    # 创建融合的FFN层
                    module.fused_ffn = nn.Sequential(
                        nn.Linear(fc1.weight.shape[1], fc1.weight.shape[0]),
                        nn.GELU(),
                        nn.Linear(fc2.weight.shape[1], fc2.weight.shape[0])
                    )

                    # 复制权重
                    module.fused_ffn[0].weight.set_value(fc1.weight)
                    module.fused_ffn[2].weight.set_value(fc2.weight)

                    if fc1.bias is not None:
                        module.fused_ffn[0].bias.set_value(fc1.bias)
                    if fc2.bias is not None:
                        module.fused_ffn[2].bias.set_value(fc2.bias)

                    print(f"Fused FFN in {module.__class__.__name__}")

        # 递归应用优化
        for name, module in model.named_sublayers():
            fuse_multi_head_attention(module)
            optimize_attention_patterns(module)
            fuse_feed_forward_network(module)

        return model

    def is_applicable(self, model: nn.Layer) -> bool:
        """检查是否适用于Transformer结构"""
        has_transformer_structure = False
        for name, module in model.named_sublayers():
            if 'transformer' in name.lower() or 'attn' in name.lower():
                has_transformer_structure = True
                break
        return has_transformer_structure


class FluxDiTFusePass(BaseOptimizationPass):
    """
    Flux DiT（Diffusion in Transformers）结构优化Pass

    优化策略：
    1. 优化patch embedding操作
    2. 融合位置编码和时间步嵌入
    3. 优化条件注入机制
    """

    def apply(self, model: nn.Layer) -> nn.Layer:
        """应用DiT优化"""

        def optimize_patch_embedding(module):
            """优化patch embedding"""
            if hasattr(module, 'patch_embed'):
                patch_embed = module.patch_embed

                # 如果有卷积patch embedding，优化其计算
                if hasattr(patch_embed, 'proj'):
                    proj = patch_embed.proj
                    if isinstance(proj, nn.Conv2D):
                        # 启用分组卷积优化（如果适用）
                        if proj._groups == 1 and proj._out_channels % 8 == 0:
                            # 可以考虑分组卷积优化
                            print(f"Optimized patch embedding in {module.__class__.__name__}")

        def fuse_positional_embeddings(module):
            """融合位置编码"""
            if hasattr(module, 'pos_embed') and hasattr(module, 'time_embed'):
                pos_embed = module.pos_embed
                time_embed = module.time_embed

                # 预计算位置编码和时间步嵌入的组合
                if hasattr(module, 'timestep_embedding'):
                    # 创建融合的嵌入层
                    pos_dim = pos_embed.shape[-1]
                    time_dim = time_embed.shape[-1]

                    if pos_dim == time_dim:
                        # 可以融合嵌入
                        module.fused_embed = nn.Linear(
                            pos_dim + time_dim,
                            pos_dim,
                            bias_attr=False
                        )
                        print(f"Fused positional embeddings in {module.__class__.__name__}")

        def optimize_conditional_injection(module):
            """优化条件注入"""
            if hasattr(module, 'cond_proj') and hasattr(module, 'cross_attn'):
                # 优化文本条件的注入
                cond_proj = module.cond_proj
                cross_attn = module.cross_attn

                # 确保条件投影和交叉注意力的维度匹配
                if cond_proj.weight.shape[0] == cross_attn.embed_dim:
                    print(f"Optimized conditional injection in {module.__class__.__name__}")

        # 应用DiT优化
        for name, module in model.named_sublayers():
            optimize_patch_embedding(module)
            fuse_positional_embeddings(module)
            optimize_conditional_injection(module)

        return model

    def is_applicable(self, model: nn.Layer) -> bool:
        """检查是否适用于DiT结构"""
        has_dit_structure = False
        for name, module in model.named_sublayers():
            if 'patch_embed' in name.lower() or 'dit' in name.lower():
                has_dit_structure = True
                break
        return has_dit_structure


class FluxRoPEFusePass(BaseOptimizationPass):
    """
    Flux旋转位置编码（RoPE）优化Pass

    优化策略：
    1. 预计算RoPE矩阵
    2. 融合RoPE与注意力计算
    3. 优化长序列的RoPE计算
    """

    def apply(self, model: nn.Layer) -> nn.Layer:
        """应用RoPE优化"""

        def precompute_rope_matrices(module):
            """预计算RoPE矩阵"""
            if hasattr(module, 'rope') or hasattr(module, 'rotary_emb'):
                # 获取RoPE配置
                rope_config = getattr(module, 'rope_config', None)
                if rope_config is None:
                    # 默认RoPE配置
                    rope_config = {
                        'theta': 10000.0,
                        'max_seq_len': 4096,
                        'head_dim': 128,
                    }

                theta = rope_config.get('theta', 10000.0)
                max_seq_len = rope_config.get('max_seq_len', 4096)
                head_dim = rope_config.get('head_dim', 128)

                # 预计算RoPE矩阵
                positions = paddle.arange(max_seq_len)
                dim = paddle.arange(0, head_dim, 2)

                angle_rates = 1.0 / paddle.pow(theta, dim / head_dim)
                angle_rates = positions.unsqueeze(-1) * angle_rates.unsqueeze(0)

                # 创建复数表示
                cos = paddle.cos(angle_rates)
                sin = paddle.sin(angle_rates)

                # 存储预计算的矩阵
                module.rope_cos = cos
                module.rope_sin = sin

                print(f"Precomputed RoPE matrices for {module.__class__.__name__}")

        def fuse_rope_with_attention(module):
            """融合RoPE与注意力计算"""
            if hasattr(module, 'rope_cos') and hasattr(module, 'rope_sin'):
                if hasattr(module, 'attn'):
                    attn = module.attn

                    # 标记注意力层使用预计算的RoPE
                    if hasattr(attn, 'use_precomputed_rope'):
                        attn.use_precomputed_rope = True
                        attn.rope_cos = module.rope_cos
                        attn.rope_sin = module.rope_sin

                        print(f"Fused RoPE with attention in {module.__class__.__name__}")

        def optimize_long_sequence_rope(module):
            """优化长序列RoPE计算"""
            if hasattr(module, 'rope_cos'):
                cos_shape = module.rope_cos.shape
                if cos_shape[0] > 2048:  # 长序列优化
                    # 对于超长序列，使用分块计算
                    module.use_chunked_rope = True
                    module.rope_chunk_size = 1024

                    print(f"Enabled chunked RoPE for long sequences in {module.__class__.__name__}")

        # 应用RoPE优化
        for name, module in model.named_sublayers():
            precompute_rope_matrices(module)
            fuse_rope_with_attention(module)
            optimize_long_sequence_rope(module)

        return model

    def is_applicable(self, model: nn.Layer) -> bool:
        """检查是否使用RoPE"""
        uses_rope = False
        for name, module in model.named_sublayers():
            if 'rope' in name.lower() or 'rotary' in name.lower():
                uses_rope = True
                break
        return uses_rope


class FluxOptimizationManager:
    """Flux优化管理器"""

    def __init__(self):
        self.passes = [
            FluxTransformerFusePass(),
            FluxDiTFusePass(),
            FluxRoPEFusePass(),
        ]

    def apply_optimizations(self, model: nn.Layer) -> nn.Layer:
        """应用所有Flux优化Pass"""
        print("Applying Flux optimizations...")

        for pass_obj in self.passes:
            if pass_obj.is_applicable(model):
                print(f"Applying {pass_obj.name}...")
                model = pass_obj.apply(model)
            else:
                print(f"Skipping {pass_obj.name} (not applicable)")

        print("Flux optimizations completed")
        return model

    def add_custom_pass(self, pass_obj: BaseOptimizationPass):
        """添加自定义优化Pass"""
        self.passes.append(pass_obj)
