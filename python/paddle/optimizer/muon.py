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
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import paddle
from paddle.base import framework
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    ShardedStateDict,
    ShardedWeight,
    create_sharded_weight_with_new_local,
)

from ..nn.clip import GradientClipBase
from .optimizer import Optimizer

# Debug logging for Muon optimizer
_logger = logging.getLogger(__name__)
MUON_DEBUG = os.environ.get("MUON_DEBUG", "0") == "1"

__all__ = []


# ------------------------------------------------------------------
# Parameter metadata dataclasses
# ------------------------------------------------------------------


@dataclass
class QKVInfo:
    """Metadata for QKV weight matrices (GQA).

    Attributes:
        head_num: Number of attention heads (Q heads).
        kv_head_num: Number of key-value heads (for GQA).
        num_key_value_groups: Number of Q heads per KV head.
    """

    head_num: int
    kv_head_num: int
    num_key_value_groups: int


@dataclass
class MLAInfo:
    """Metadata for MLA weight matrices needed for head-split.

    Attributes:
        param_name: Name of the parameter (q_b_proj, kv_b_proj, o_proj).
        head_num: Number of attention heads.
    """

    param_name: str
    head_num: int


@dataclass
class MuonParamInfo:
    """Muon update metadata for a single parameter.

    This replaces the previous approach of setting dynamic attributes
    directly on param objects.

    Attributes:
        use_muon: If True, use Muon (orthogonal) updates; otherwise AdamW.
        qkv_info: Required for QKV weight matrices.
        intermediate_size: Required for FFN gate_up weights when muon_ffn_split is True.
    """

    use_muon: bool = True
    qkv_info: QKVInfo | None = None
    mla_info: MLAInfo | None = None
    intermediate_size: int | None = None

    @property
    def is_qkv(self) -> bool:
        """True if this is a QKV weight matrix."""
        return self.qkv_info is not None

    @property
    def is_mla(self) -> bool:
        """True if this is an MLA weight matrix."""
        return self.mla_info is not None

    @property
    def is_ffn_gate_up(self) -> bool:
        """True if this is an FFN gate_up weight matrix."""
        return self.intermediate_size is not None


# Type alias for the parameter info mapping
MuonParamInfoMap = dict[str, MuonParamInfo]

# ------------------------------------------------------------------
# Newton-Schulz coefficient sets
# ------------------------------------------------------------------

_NS_COEFFICIENT_SETS = {
    # Simple coefficient set (original)
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    # Quintic iteration with optimized coefficients
    # Source: https://leloykun.github.io/ponder/muon-opt-coeffs/
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    # Polar Express iteration from https://arxiv.org/abs/2505.16932
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    # AOL coefficients from https://github.com/thib-s/flash-newton-schulz
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}

# ------------------------------------------------------------------
# Default parameter classification
# ------------------------------------------------------------------


def _default_should_use_muon(name, shape, exclude_patterns):
    """Default fallback logic for determining if a parameter should use Muon.

    This is only used when param.is_muon is not set. The actual exclusion
    patterns must be configured via training_args.muon_exclude_patterns in yaml.

    Args:
        name: Parameter name.
        shape: Parameter shape.
        exclude_patterns: List of substrings to exclude from Muon updates.
            Must be provided (e.g., ['embed', 'bias', 'lm_head', 'mlp.gate']).

    Returns:
        True if the parameter should use Muon (orthogonal) updates.

    Raises:
        ValueError: If exclude_patterns is None.
    """
    if exclude_patterns is None:
        raise ValueError(
            "muon_exclude_patterns must be set in yaml config. "
            "Example: muon_exclude_patterns: ['embed', 'bias', 'lm_head', 'mlp.gate']"
        )

    if len(shape) not in (2, 3):
        return False

    name_lower = name.lower()
    for pattern in exclude_patterns:
        if pattern.lower() in name_lower:
            return False
    return True


class Muon(Optimizer):
    r"""
    Muon optimizer for MuonShardingOptimizer (Sharding Stage1 V3) usage.

    For 2-D weight matrices (identified by :func:`_default_should_use_muon`), Muon
    applies orthogonal gradient updates via Newton-Schulz iteration.  For all
    other parameters (embeddings, biases, expert weights, …) it falls back to
    a standard AdamW update.

    Designed for ``MuonShardingOptimizer`` (Sharding Stage1 V3), where 2D parameters are
    assigned as whole tensors to ranks. Currently we do not support TP=1, no sharding gather
    or TP communication is needed during the optimizer step.

    Args:
        learning_rate (float | LRScheduler): Learning rate. Default: ``0.02``.
        parameters (list[Tensor]): Flat list of parameters to optimize.
        momentum (float): Momentum coefficient for the Muon update. Default: ``0.95``.
        adam_beta1 (float): β₁ for the AdamW fallback. Default: ``0.9``.
        adam_beta2 (float): β₂ for the AdamW fallback. Default: ``0.95``.
        weight_decay (float): Decoupled weight decay. Default: ``0.01``.
        ns_steps (int): Newton-Schulz iteration steps. Default: ``5``.
        nesterov (bool): Use Nesterov momentum in Muon. Default: ``True``.
        adam_epsilon (float): ε for numerical stability in AdamW. Default: ``1e-9``.
        grad_clip (GradientClipBase | None): Gradient clipping. Default: ``None``.
        apply_decay_param_fun (callable | None): Function to select which
            parameters receive weight decay. Default: ``None``.
        muon_version (int): Scaling-function version (1/2/3). Default: ``1``.
        muon_exclude_patterns (list[str] | None): Parameter names containing
            any of these substrings will use AdamW instead of Muon.
            Example: ``['embed', 'bias', 'lm_head', 'mlp.gate']``.
            Default: ``None``.
        muon_qkv_update_mode (str): Strategy for QKV fused weight matrices.
            ``"split_head"`` orthogonalises each Q/K/V head independently;
            ``"split_qkv"`` treats Q, K, V as three separate matrices;
            ``"fused_qkv"`` treats the entire QKV matrix as one.
            Default: ``"split_head"``.
        muon_ffn_split (bool): If True, split FFN gate_up fused weights into
            gate and up projections and orthogonalise them independently.
            Default: ``False``.
        muon_extra_scale_factor (float): Extra multiplicative scale applied
            after the dimension-dependent scaling in ``_scaling_fn``.
            Default: ``0.2``.
        muon_param_info_map (MuonParamInfoMap | None): Per-parameter metadata
            dict mapping param name to :class:`MuonParamInfo` (use_muon,
            qkv_info, intermediate_size). Built by Trainer and passed in.
            Default: ``None``.
        multi_precision (bool): Maintain FP32 master weights when training in
            BF16/FP16. Default: ``False``.
        name (str | None): Optional name for the optimizer instance.
    """

    _moment_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(
        self,
        learning_rate=0.02,
        parameters=None,
        momentum=0.95,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        ns_steps=5,
        ns_coeff_type="simple",
        nesterov=True,
        adam_epsilon=1e-9,
        grad_clip=None,
        apply_decay_param_fun=None,
        muon_version=1,
        muon_exclude_patterns=None,
        muon_qkv_update_mode="split_head",
        muon_ffn_split=False,
        muon_extra_scale_factor=0.2,
        muon_param_info_map: MuonParamInfoMap | None = None,
        multi_precision=False,
        name=None,
        **kwargs,
    ):
        if parameters is None:
            raise ValueError(
                "parameters argument given to the Optimizer should not be None."
            )
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a list.")
        if len(parameters) > 0 and isinstance(parameters[0], dict):
            raise TypeError(
                "Muon optimizer only supports a flat list of parameters, "
                "not a list of parameter groups."
            )
        if grad_clip is not None and not isinstance(
            grad_clip, GradientClipBase
        ):
            raise TypeError(
                "'grad_clip' should be an instance of GradientClipBase's derived class"
            )

        defaults = {
            "momentum": momentum,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "nesterov": nesterov,
            "epsilon": adam_epsilon,
            "muon_version": muon_version,
            "ns_coeff_type": ns_coeff_type,
        }

        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )

        self._multi_precision = multi_precision
        self._master_weights = {}
        self._apply_decay_param_fun = apply_decay_param_fun
        self._muon_split_logged = False
        self._muon_exclude_patterns = muon_exclude_patterns
        self._muon_qkv_update_mode = muon_qkv_update_mode
        self._muon_ffn_split = muon_ffn_split
        self._muon_extra_scale_factor = muon_extra_scale_factor
        self._ns_coeff_type = ns_coeff_type
        self._muon_param_info_map = muon_param_info_map or {}
        self._default_dict.update(defaults)

    # ------------------------------------------------------------------
    # Accumulator management
    # ------------------------------------------------------------------

    def _ensure_accumulators(self, param, use_muon, group):
        """Create optimizer accumulators for *param* if they do not exist yet."""
        if (
            self._moment_acc_str in self._accumulators
            and param.name in self._accumulators[self._moment_acc_str]
        ):
            return

        # FP32 master weight for mixed-precision training
        if self._multi_precision and self._is_dtype_fp16_or_bf16(param.dtype):
            if param.name not in self._master_weights:
                self._create_master_weight(param)

        self._add_accumulator(
            self._moment_acc_str,
            param,
            dtype=paddle.float32,
            fill_value=0.0,
            shape=param.shape,
            type=framework.core.VarDesc.VarType.DENSE_TENSOR,
        )

        if not use_muon:
            # AdamW-specific states
            self._add_accumulator(
                self._moment2_acc_str,
                param,
                dtype=paddle.float32,
                fill_value=0.0,
                shape=param.shape,
                type=framework.core.VarDesc.VarType.DENSE_TENSOR,
            )
            for acc_name, init_val in [
                (self._beta1_pow_acc_str, group.get("adam_beta1", 0.9)),
                (self._beta2_pow_acc_str, group.get("adam_beta2", 0.95)),
            ]:
                self._add_accumulator(
                    acc_name,
                    param,
                    dtype=paddle.float32,
                    fill_value=1.0,
                    shape=[1],
                    type=framework.core.VarDesc.VarType.DENSE_TENSOR,
                )

    def _create_accumulators(self, block, parameters):
        """Standard entry-point used by checkpoint-resume infrastructure."""
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)
        for p in parameters:
            param_info = self._muon_param_info_map.get(p.name)
            if param_info is not None:
                use_muon = param_info.use_muon
            else:
                use_muon = _default_should_use_muon(
                    p.name,
                    getattr(p, "original_shape", p.shape),
                    self._muon_exclude_patterns,
                )
            self._ensure_accumulators(p, use_muon, self._default_dict)

    # ------------------------------------------------------------------
    # Newton-Schulz orthogonalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _zeropower_via_newtonschulz5(
        X, steps=5, eps=1e-9, ns_coeff_type="simple"
    ):
        """Approximate the matrix sign function via Newton-Schulz iteration.

        Args:
            X: Input tensor to orthogonalize.
            steps: Number of Newton-Schulz iterations.
            eps: Small constant for numerical stability.
            ns_coeff_type: Type of coefficient set to use.
                Options: "simple", "quintic", "polar_express", "aol".
        """
        # Get coefficient set
        coeff_sets = _NS_COEFFICIENT_SETS.get(
            ns_coeff_type, _NS_COEFFICIENT_SETS["simple"]
        )

        if X.shape[-2] > X.shape[-1]:
            X = X.T
            transpose = True
        else:
            transpose = False

        orig_shape = X.shape
        X_flat = X.flatten(start_axis=-2)
        X_flat = paddle.nn.functional.normalize(
            X_flat, p=2, axis=-1, epsilon=eps
        )
        X = X_flat.reshape(orig_shape).astype(paddle.bfloat16)

        # Iterate with cycling coefficients
        for i in range(steps):
            a, b, c = coeff_sets[i % len(coeff_sets)]
            A = paddle.matmul(X, X, transpose_y=True)
            B = paddle.addmm(input=A, x=A, y=A, beta=b, alpha=c)
            X = paddle.addmm(input=X, x=B, y=X, beta=a, alpha=1.0)

        return X.T if transpose else X

    @staticmethod
    def _scaling_fn(orthogonal_update, version, extra_scale_factor=1.0):
        """Apply dimension-dependent scaling to the orthogonal update."""
        din, dout = orthogonal_update.shape[0], orthogonal_update.shape[1]
        if version == 1:
            scale = max(1, dout / din) ** 0.5
        elif version == 2:
            scale = (dout / din) ** 0.5
        else:  # version == 3 (default)
            scale = max(dout, din) ** 0.5
        return orthogonal_update * scale * extra_scale_factor

    @staticmethod
    def _ortho_qkv_per_head(
        matrix_2d_global,
        kv_head_num,
        num_key_value_groups,
        ortho_fn,
    ):
        """Orthogonalise each Q/K/V head independently (interleaved layout).

        Args:
            matrix_2d_global: QKV weight matrix [hidden, (num_key_value_groups + 2)*kv_head_num*head_dim].
            kv_head_num: Number of K/V heads.
            num_key_value_groups: Number of Q heads per KV head.
            ortho_fn: Callable (2d_matrix) -> 2d_matrix applying NS + scaling.

        Returns:
            orthogonal_update: Same shape as input, each head orthogonalised.
        """
        # Interleaved layout: [Q_kv0, K0, V0, Q_kv1, K1, V1, ...]
        head_dim = matrix_2d_global.shape[1] // (
            num_key_value_groups * kv_head_num + 2 * kv_head_num
        )
        groups = paddle.split(matrix_2d_global, kv_head_num, axis=1)

        processed_groups = []
        for group in groups:
            q_part, k_head, v_head = paddle.split(
                group,
                [num_key_value_groups * head_dim, head_dim, head_dim],
                axis=1,
            )
            q_heads = paddle.split(q_part, num_key_value_groups, axis=1)
            q_ortho = paddle.concat([ortho_fn(h) for h in q_heads], axis=1)
            processed_groups.append(
                paddle.concat(
                    [q_ortho, ortho_fn(k_head), ortho_fn(v_head)], axis=1
                )
            )

        return paddle.concat(processed_groups, axis=1)

    @staticmethod
    def _ortho_qkv_sep(
        matrix_2d,
        kv_head_num,
        num_key_value_groups,
        ortho_fn,
    ):
        """Orthogonalise Q, K, V as three separate whole matrices (interleaved layout).

        Gathers all Q heads into one block, all K heads into one block, all V heads
        into one block (across kv_groups), orthogonalises each block as a whole with
        one NS call, then scatters back to interleaved order.

        Args:
            matrix_2d: QKV weight matrix [hidden, (num_key_value_groups + 2)*kv_head_num*head_dim].
            kv_head_num: Number of K/V heads.
            num_key_value_groups: Number of Q heads per KV head.
            ortho_fn: Callable (2d_matrix) -> 2d_matrix applying NS + scaling.

        Returns:
            orthogonal_update: Same shape as input, Q/K/V each orthogonalised as whole.
        """
        # Interleaved layout: [Q_kv0, K0, V0, Q_kv1, K1, V1, ...]
        head_dim = matrix_2d.shape[1] // (
            num_key_value_groups * kv_head_num + 2 * kv_head_num
        )
        q_group_size = num_key_value_groups * head_dim

        # Step 1: gather Q / K / V parts from each kv_group
        groups = paddle.split(matrix_2d, kv_head_num, axis=1)
        q_parts, k_parts, v_parts = [], [], []
        for group in groups:
            q_p, k_p, v_p = paddle.split(
                group, [q_group_size, head_dim, head_dim], axis=1
            )
            q_parts.append(q_p)
            k_parts.append(k_p)
            v_parts.append(v_p)

        # Step 2: orthogonalise each projection as one whole matrix
        q_ortho = ortho_fn(paddle.concat(q_parts, axis=1))
        k_ortho = ortho_fn(paddle.concat(k_parts, axis=1))
        v_ortho = ortho_fn(paddle.concat(v_parts, axis=1))

        # Step 3: split back and restore interleaved layout
        q_groups = paddle.split(q_ortho, kv_head_num, axis=1)
        k_groups = paddle.split(k_ortho, kv_head_num, axis=1)
        v_groups = paddle.split(v_ortho, kv_head_num, axis=1)

        return paddle.concat(
            [
                paddle.concat([q_groups[i], k_groups[i], v_groups[i]], axis=1)
                for i in range(kv_head_num)
            ],
            axis=1,
        )

    @staticmethod
    def _ortho_ffn_gate_up(matrix, intermediate_size, ortho_fn):
        """Orthogonalise gate and up projections independently for FFN.

        Args:
            matrix: FFN weight tensor.
                - 2D: [hidden, 2*intermediate_size] for standard FFN
                - 3D: [num_experts, hidden, 2*intermediate_size] for MoE FFN
            intermediate_size: Size of each of gate/up projections.
            ortho_fn: Callable (2d_matrix) -> 2d_matrix applying NS + scaling.

        Returns:
            orthogonal_update: Tensor with gate and up orthogonalised separately.
        """
        if matrix.ndim == 2:
            gate, up = paddle.split(
                matrix, [intermediate_size, intermediate_size], axis=1
            )
            return paddle.concat([ortho_fn(gate), ortho_fn(up)], axis=1)

        elif matrix.ndim == 3:
            # MoE FFN: [n_experts, hidden, 2*intermediate_size]
            expert_updates = []
            for ei in range(matrix.shape[0]):
                gate, up = paddle.split(
                    matrix[ei], [intermediate_size, intermediate_size], axis=1
                )
                expert_updates.append(
                    paddle.concat([ortho_fn(gate), ortho_fn(up)], axis=1)
                )
            return paddle.stack(expert_updates, axis=0)

        else:
            raise ValueError(
                f"FFN gate_up split expects 2D or 3D tensor, got shape {matrix.shape}"
            )

    @staticmethod
    def _ortho_mla_per_head(
        matrix_2d_global,
        head_num,
        ortho_fn,
        axis,
    ):
        """Orthogonalise each MLA head independently."""
        groups = paddle.split(matrix_2d_global, head_num, axis=axis)

        processed_groups = []
        for group in groups:
            processed_groups.append(ortho_fn(group))

        return paddle.concat(processed_groups, axis=axis)

    # ------------------------------------------------------------------
    # Per-parameter update rules
    # ------------------------------------------------------------------

    @staticmethod
    def _adamw_update(
        param,
        grad,
        lr,
        moment1,
        moment2,
        beta1_pow,
        beta2_pow,
        beta1,
        beta2,
        epsilon,
        weight_decay,
    ):
        """In-place AdamW update for 1-D sharded parameters."""
        with paddle.no_grad():
            beta1_pow.scale_(beta1)
            beta2_pow.scale_(beta2)

            if weight_decay > 0:
                param.scale_(1.0 - lr * weight_decay)

            grad_f32 = (
                grad.astype(paddle.float32)
                if grad.dtype != paddle.float32
                else grad
            )

            moment1.scale_(beta1).add_(grad_f32, alpha=1.0 - beta1)
            moment2.scale_(beta2).add_(
                paddle.square(grad_f32), alpha=1.0 - beta2
            )

            bias1 = 1.0 - beta1_pow
            bias2 = 1.0 - beta2_pow
            update = (
                (moment1 / bias1)
                / ((paddle.sqrt(moment2) / paddle.sqrt(bias2)) + epsilon)
                * lr
            )

            if update.dtype != param.dtype:
                update = update.astype(param.dtype)

            if hasattr(param, "subtract_"):
                param.subtract_(update)
            else:
                paddle.assign(param - update, param)

    def _muon_update(
        self,
        param,
        grad,
        lr,
        momentum_buffer,
        momentum_beta,
        ns_steps,
        nesterov,
        epsilon,
        weight_decay,
        version,
    ):
        """In-place Muon update for a 2D parameter tensor.

        Applies Newton-Schulz orthogonalisation to the 2D weight matrix and
        updates the parameter in-place. MuonShardingOptimizer assigns whole
        2D tensors to ranks, so no sharding gather or TP communication is needed.
        """
        param_shape = getattr(param, "original_shape", param.shape)
        param_info = self._muon_param_info_map.get(param.name)
        is_qkv = param_info is not None and param_info.is_qkv
        is_mla: bool = param_info is not None and param_info.is_mla
        is_ffn_gate_up = param_info is not None and param_info.is_ffn_gate_up

        with paddle.no_grad():
            grad_f32 = (
                grad.astype(momentum_buffer.dtype)
                if grad.dtype != momentum_buffer.dtype
                else grad
            )

            # Step 1: Momentum update
            new_momentum = paddle.lerp(
                momentum_buffer, grad_f32, 1.0 - momentum_beta
            )
            paddle.assign(new_momentum, momentum_buffer)
            update_buffer = (
                paddle.lerp(grad_f32, momentum_buffer, momentum_beta)
                if nesterov
                else momentum_buffer
            )

            # Step 2: Reshape update buffer to 2D matrix.
            # MuonShardingOptimizer assigns whole 2D tensors to ranks, so params
            # are already 2D/3D (no sharding gather needed).
            matrix_2d_global = update_buffer.reshape(param_shape)

            # Shared NS + scaling closure (captures ns_steps, epsilon, version, ns_coeff_type)
            def ortho_fn(m):
                ns_out = Muon._zeropower_via_newtonschulz5(
                    m,
                    steps=ns_steps,
                    eps=epsilon,
                    ns_coeff_type=self._ns_coeff_type,
                )
                scaled = Muon._scaling_fn(
                    ns_out, version, self._muon_extra_scale_factor
                )
                return scaled

            # Step 3: Newton-Schulz orthogonalisation
            if is_ffn_gate_up and self._muon_ffn_split:
                # FFN gate_up split: orthogonalise gate and up projections independently.
                intermediate_size = param_info.intermediate_size
                if MUON_DEBUG:
                    _global_rank = paddle.distributed.get_rank()
                    if _global_rank == 0:
                        _logger.info(
                            f"[Muon] FFN split: param={param.name}, "
                            f"shape={matrix_2d_global.shape}, "
                            f"intermediate_size={intermediate_size}"
                        )

                orthogonal_update = Muon._ortho_ffn_gate_up(
                    matrix_2d_global, intermediate_size, ortho_fn
                )
            elif matrix_2d_global.ndim == 3:
                # 3D fused MoE expert tensor [n_experts, H, I].
                # Apply Newton-Schulz independently to each expert's 2D slice.
                n_experts = matrix_2d_global.shape[0]
                orthogonal_update = paddle.stack(
                    [ortho_fn(matrix_2d_global[ei]) for ei in range(n_experts)],
                    axis=0,
                )
            elif is_qkv and self._muon_qkv_update_mode in (
                "split_head",
                "split_qkv",
            ):
                # Read QKV head info from param_info
                qkv_info = param_info.qkv_info
                kv_head_num = qkv_info.kv_head_num
                num_key_value_groups = qkv_info.num_key_value_groups

                if self._muon_qkv_update_mode == "split_head":
                    # split_head update: each Q/K/V head orthogonalised independently.
                    if MUON_DEBUG:
                        _global_rank = paddle.distributed.get_rank()
                        if _global_rank == 0:
                            _logger.info(
                                f"[Muon] QKV split_head: param={param.name}, "
                                f"shape={matrix_2d_global.shape}, "
                                f"heads={qkv_info.head_num}/{kv_head_num}, "
                                f"num_key_value_groups={num_key_value_groups}"
                            )
                    orthogonal_update = Muon._ortho_qkv_per_head(
                        matrix_2d_global,
                        kv_head_num,
                        num_key_value_groups,
                        ortho_fn,
                    )
                else:
                    # split_qkv: Q, K, V each as a whole matrix, one NS call each.
                    if MUON_DEBUG:
                        _global_rank = paddle.distributed.get_rank()
                        if _global_rank == 0:
                            _logger.info(
                                f"[Muon] QKV split_qkv: param={param.name}, "
                                f"shape={matrix_2d_global.shape}, "
                                f"head_num={qkv_info.head_num}, kv_head_num={kv_head_num}, "
                                f"num_key_value_groups={num_key_value_groups}"
                            )
                    orthogonal_update = Muon._ortho_qkv_sep(
                        matrix_2d_global,
                        kv_head_num,
                        num_key_value_groups,
                        ortho_fn,
                    )
            elif is_mla and self._muon_qkv_update_mode == "split_head":
                # MLA split_head update: each head of [q_b_proj, kv_b_proj, o_proj] orthogonalised independently.
                mla_info = param_info.mla_info
                param_name: str = mla_info.param_name
                head_num = mla_info.head_num
                if MUON_DEBUG:
                    _global_rank = paddle.distributed.get_rank()
                    if _global_rank == 0:
                        _logger.info(
                            f"[Muon] MLA split_head: param={param.name}, param_name={param_name}, "
                            f"shape={matrix_2d_global.shape}, "
                            f"head_num={head_num}"
                        )
                assert param_name in ("q_b_proj", "kv_b_proj", "o_proj"), (
                    f"Unsupported MLA param name: {param_name}"
                )
                orthogonal_update = Muon._ortho_mla_per_head(
                    matrix_2d_global,
                    head_num,
                    ortho_fn,
                    0 if param_name == "o_proj" else 1,
                )
            else:
                # Standard 2D update: entire matrix as one Newton-Schulz call.
                orthogonal_update = ortho_fn(matrix_2d_global)

            # Step 4: Apply update with optional weight decay
            if weight_decay > 0:
                param.scale_(1.0 - lr * weight_decay)

            final_step = orthogonal_update * lr
            if final_step.dtype != param.dtype:
                final_step = final_step.astype(param.dtype)

            if hasattr(param, "subtract_"):
                param.subtract_(final_step)
            else:
                paddle.assign(param - final_step, param)

    # ------------------------------------------------------------------
    # Core optimization step
    # ------------------------------------------------------------------

    def _apply_optimize(self, loss, startup_program, params_grads):
        if not framework.in_dygraph_mode():
            raise NotImplementedError(
                "Muon optimizer only supports dygraph mode."
            )

        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)

        group = self._default_dict
        lr = self._learning_rate
        if isinstance(lr, paddle.optimizer.lr.LRScheduler):
            lr = lr()
        wd = group.get("weight_decay", 0.0)

        muon_params = []
        adamw_params = []
        for param, grad in params_grads:
            if grad is None:
                continue

            param_info = self._muon_param_info_map.get(param.name)
            assert param_info is not None, (
                f"muon_param_info_map does not have {param.name}"
            )
            use_muon = param_info.use_muon

            self._ensure_accumulators(param, use_muon, group)
            if use_muon:
                muon_params.append((param, grad))
            else:
                adamw_params.append((param, grad))

        # --- Pass 1: Muon updates (large temporary tensors) ---
        for param, grad in muon_params:
            self._muon_update(
                param,
                grad,
                lr,
                self._get_accumulator(self._moment_acc_str, param),
                group.get("momentum", 0.95),
                group.get("ns_steps", 5),
                group.get("nesterov", True),
                group.get("epsilon", 1e-9),
                wd,
                version=group.get("muon_version", 3),
            )
            if self._multi_precision and param.name in self._master_weights:
                with paddle.no_grad():
                    _cast_tmp = paddle.cast(param, paddle.float32)
                    paddle.assign(_cast_tmp, self._master_weights[param.name])
                    del _cast_tmp

        # --- Pass 2: AdamW updates ---
        for param, grad in adamw_params:
            self._adamw_update(
                param,
                grad,
                lr,
                self._get_accumulator(self._moment_acc_str, param),
                self._get_accumulator(self._moment2_acc_str, param),
                self._get_accumulator(self._beta1_pow_acc_str, param),
                self._get_accumulator(self._beta2_pow_acc_str, param),
                group.get("adam_beta1", 0.9),
                group.get("adam_beta2", 0.95),
                group.get("epsilon", 1e-9),
                wd,
            )
            if self._multi_precision and param.name in self._master_weights:
                with paddle.no_grad():
                    _cast_tmp = paddle.cast(param, paddle.float32)
                    paddle.assign(_cast_tmp, self._master_weights[param.name])
                    del _cast_tmp

    @framework.dygraph_only
    def step(self) -> None:
        params_grads = [
            (param, param._grad_ivar())
            for param in self._parameter_list
            if not param.stop_gradient and param._grad_ivar() is not None
        ]
        self._apply_optimize(
            loss=None, startup_program=None, params_grads=params_grads
        )

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
    ) -> ShardedStateDict:
        """Build a sharded optimizer state dict for flex checkpoint save/load.

        The layout mirrors :class:`paddle.optimizer.AdamW`'s implementation so
        that the same ``dist.save_state_dict`` / ``dist.load_state_dict`` path
        works for Muon checkpoints.

        Args:
            model_sharded_state_dict: Sharded model state dict produced by
                ``model.sharded_state_dict()``.

        Returns:
            A dict mapping ``"<struct_name>.<state_type>"`` keys to
            :class:`ShardedWeight` objects.
        """
        _FP32_MASTER = "fp32_master_0"
        _optimizer_scalar_names = [
            "beta1_pow_acc_0",
            "beta2_pow_acc_0",
        ]
        _optimizer_vector_names = [
            "moment1_0",
            "moment2_0",
        ]

        def _split_state_name(vname):
            if _FP32_MASTER in vname:
                return tuple(vname.split("_" + _FP32_MASTER + "_", 1))
            for suffix in _optimizer_scalar_names + _optimizer_vector_names:
                if vname.endswith(suffix):
                    return vname[: -(len(suffix) + 1)], suffix
            raise ValueError(
                f"Cannot parse optimizer state variable name: {vname!r}"
            )

        model_sharded_state_dict = dict(
            sorted(model_sharded_state_dict.items())
        )

        # Build static-name → struct-name mapping (handles shared weights)
        static_to_struct = {}
        for struct_name, sw in model_sharded_state_dict.items():
            local_name = sw.local_tensor.name
            if local_name not in static_to_struct:
                static_to_struct[local_name] = struct_name

        optimizer_state_dict = self.state_dict()
        master_weights = optimizer_state_dict.pop("master_weights", None)
        optimizer_state_dict.pop("LR_Scheduler", None)

        sharded_state: ShardedStateDict = {}

        # Optimizer states (moment1, moment2, beta_pow scalars)
        for key, tensor in optimizer_state_dict.items():
            static_name, state_type = _split_state_name(key)
            struct_name = static_to_struct[static_name]
            sharded_param = model_sharded_state_dict[struct_name]
            unified_name = f"{struct_name}.{state_type}"

            if state_type in _optimizer_vector_names:
                # Vector states share the same sharding layout as the parameter
                if tensor.is_dist():
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=tensor,
                        local_shape=tensor.shape,
                        global_shape=tensor.shape,
                        global_offset=sharded_param.global_offset,
                    )
                else:
                    # Reshape accumulator if numel matches but shape differs.
                    # MoE: grouped_gemm_experts param.shape is 3D
                    # [n_experts, H, I] but model.state_dict() returns actual
                    # C++ storage shape 2D [n_experts*H, I].  moment1 was
                    # created with 3D shape, so we need to reshape here.
                    # V2 is unaffected: its moments are always 1D shards,
                    # so shape always matches and reshape is never triggered.
                    target_shape = sharded_param.local_shape
                    if (
                        tuple(tensor.shape) != tuple(target_shape)
                        and tensor.numel()
                        == paddle.to_tensor(list(target_shape)).prod().item()
                    ):
                        tensor = tensor.reshape(target_shape)
                    sharded_state[unified_name] = (
                        create_sharded_weight_with_new_local(
                            unified_name, tensor, sharded_param
                        )
                    )
            else:
                # Scalar states (beta_pow) are replicated – save as-is
                sharded_state[unified_name] = ShardedWeight(
                    key=unified_name,
                    local_tensor=tensor,
                    local_shape=(1,),
                    global_shape=(1,),
                    global_offset=(0,),
                )

        # FP32 master weights
        if master_weights:
            for weight_key, tensor in master_weights.items():
                struct_name = static_to_struct[weight_key]
                sharded_param = model_sharded_state_dict[struct_name]
                unified_name = f"{struct_name}.w_0"

                if tensor.is_dist():
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=tensor,
                        local_shape=tensor.shape,
                        global_shape=tensor.shape,
                        global_offset=sharded_param.global_offset,
                    )
                else:
                    sharded_state[unified_name] = (
                        create_sharded_weight_with_new_local(
                            unified_name, tensor, sharded_param
                        )
                    )

        return sharded_state
