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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle import Tensor

import paddle
from paddle import _C_ops
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
class MuonParamInfo:
    """Muon update metadata for a single parameter.

    This replaces the previous approach of setting dynamic attributes
    directly on param objects.

    Attributes:
        use_muon: If True, use Muon (orthogonal) updates; otherwise AdamW.
        split_concat_func: Optional callable that implements the slice strategy.
            Signature: split_concat_func(matrix, ortho_fn, **kwargs) -> sliced_matrix
            If None, whole-matrix orthogonalisation is used.
    """

    use_muon: bool = True
    split_concat_func: Callable | None = None


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
    "deepseekv4":
    # From DeepSeekV4: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf
    [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2,
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
        ns_coeff_type (str): Preset name for Newton-Schulz coefficients.
            Options: ``"simple"``, ``"quintic"``, ``"polar_express"``,
            ``"aol"``, ``"deepseekv4"``, ``"custom"``. Default: ``"simple"``.
        ns_coeffs (list[tuple[float, float, float]] | None): Custom
            Newton-Schulz coefficient set. Each tuple is ``(a, b, c)``
            for one iteration step. Default: ``None``.
            Only used when ns_coeff_type=``custom``.
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
        muon_extra_scale_factor (float): Extra multiplicative scale applied
            after the dimension-dependent scaling in ``_scaling_fn``.
            Default: ``0.2``.
        muon_param_info_map (MuonParamInfoMap | None): Per-parameter metadata
            dict mapping param name to :class:`MuonParamInfo` (use_muon,
            split_concat_func). Built by Trainer and passed in.
            Default: ``None``.
        ns_matmul_dtype (paddle.dtype | None): Dtype for Newton-Schulz matmul
            iterations. ``None`` = auto-detect: bfloat16 on Ampere+ (capability
            >= 8.0), float32 on V100 and older. Pass ``paddle.float32``
            explicitly to force float32. Default: ``None``.
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
        ns_coeffs=None,
        nesterov=True,
        adam_epsilon=1e-9,
        grad_clip=None,
        lr_ratio: Callable[[Tensor], float] | None = None,
        apply_decay_param_fun: Callable[[str], bool] | None = None,
        muon_version=1,
        muon_exclude_patterns=None,
        muon_extra_scale_factor=0.2,
        muon_param_info_map: MuonParamInfoMap | None = None,
        ns_matmul_dtype=None,
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
        self._lr_ratio = lr_ratio
        self._apply_decay_param_fun = apply_decay_param_fun
        self._muon_split_logged = False
        self._muon_exclude_patterns = muon_exclude_patterns
        self._muon_extra_scale_factor = muon_extra_scale_factor
        self._ns_coeff_type = ns_coeff_type
        if ns_coeff_type == "custom":
            assert ns_coeffs is not None, (
                "ns_coeffs must be provided when ns_coeff_type is 'custom'."
            )
            self._ns_coeffs = ns_coeffs
        else:
            assert ns_coeff_type in _NS_COEFFICIENT_SETS, (
                f"Invalid ns_coeff_type: {ns_coeff_type}"
            )
            self._ns_coeffs = _NS_COEFFICIENT_SETS[ns_coeff_type]
        self._muon_param_info_map = muon_param_info_map or {}
        # Dtype for Newton-Schulz matmul.
        # None = auto: bfloat16 on Ampere+ (capability >= 8.0), float32 on older.
        if ns_matmul_dtype is None:
            cap = (
                paddle.device.cuda.get_device_capability()
                if paddle.is_compiled_with_cuda()
                else (0, 0)
            )
            self._ns_matmul_dtype = (
                paddle.bfloat16 if cap[0] >= 8 else paddle.float32
            )
        else:
            self._ns_matmul_dtype = ns_matmul_dtype
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
                    fill_value=init_val,
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
        X,
        steps=5,
        eps=1e-9,
        ns_coeffs=None,
        ns_matmul_dtype=paddle.bfloat16,
    ):
        """Approximate the matrix sign function via Newton-Schulz iteration.

        Args:
            X: Input tensor to orthogonalize. Must be 2D (M, N) or
                3D (B, M, N) for batched operation.
            steps: Number of Newton-Schulz iterations.
            eps: Small constant for numerical stability.
            ns_coeffs: List of (a, b, c) coefficient tuples for iteration.
                If None, uses the "simple" preset.
            ns_matmul_dtype: Dtype for matmul iterations. Defaults to
                bfloat16. Pass paddle.float32 for V100 compatibility.
        """
        if X.ndim < 2 or X.ndim > 3:
            raise ValueError(
                f"Input tensor X must be 2D or 3D (batched), got {X.ndim}D"
            )

        coeff_sets = (
            ns_coeffs
            if ns_coeffs is not None
            else _NS_COEFFICIENT_SETS["simple"]
        )

        if X.shape[-2] > X.shape[-1]:
            X = paddle.transpose(
                X,
                perm=[1, 0] if X.ndim == 2 else [0, 2, 1],
            )
            transpose = True
        else:
            transpose = False

        orig_shape = X.shape
        X_flat = X.flatten(start_axis=-2)
        X_flat = paddle.nn.functional.normalize(
            X_flat, p=2, axis=-1, epsilon=eps
        )
        X = X_flat.reshape(orig_shape).astype(ns_matmul_dtype)

        if X.ndim == 3:
            ns_step_fn = Muon._batched_newton_schulz_step
        else:
            ns_step_fn = Muon._newton_schulz_step

        for i in range(steps):
            a, b, c = coeff_sets[i % len(coeff_sets)]
            X = ns_step_fn(X, a, b, c)

        if transpose:
            X = paddle.transpose(X, perm=[1, 0] if X.ndim == 2 else [0, 2, 1])
        return X

    @staticmethod
    def _newton_schulz_step(X, a, b, c):
        """Single Newton-Schulz iteration step for 2D input."""
        A = paddle.matmul(X, X, transpose_y=True)
        B = paddle.addmm(input=A, x=A, y=A, beta=b, alpha=c)
        X = paddle.addmm(input=X, x=B, y=X, beta=a, alpha=1.0)
        return X

    @staticmethod
    def _batched_newton_schulz_step(X, a, b, c):
        """Single Newton-Schulz iteration step for 3D batched input."""
        A = paddle.matmul(X, X, transpose_y=True)
        B = paddle.baddbmm(A, A, A, beta=b, alpha=c)
        X = paddle.baddbmm(X, B, X, beta=a, alpha=1.0)
        return X

    @staticmethod
    def _scaling_fn(orthogonal_update, version, extra_scale_factor=1.0):
        """Apply dimension-dependent scaling to the orthogonal update."""
        din, dout = orthogonal_update.shape[-2], orthogonal_update.shape[-1]
        if version == 1:
            scale = max(1, dout / din) ** 0.5
        elif version == 2:
            scale = (dout / din) ** 0.5
        else:  # version == 3 (default)
            scale = max(dout, din) ** 0.5
        return orthogonal_update * scale * extra_scale_factor

    # ------------------------------------------------------------------
    # Per-parameter update rules
    # ------------------------------------------------------------------

    def _adamw_update(
        self,
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

        lr_ratio = 1.0 if self._lr_ratio is None else self._lr_ratio(param)
        with_decay = True
        if (
            self._apply_decay_param_fun is not None
            and not self._apply_decay_param_fun(param.name)
        ):
            with_decay = False

        find_master = param.name in self._master_weights
        master_weight = (
            self._master_weights[param.name] if find_master else None
        )
        _, _, _, _, _, _, _ = _C_ops.adamw_(
            param,
            grad,
            lr,
            moment1,
            moment2,
            None,  # moment2_max
            beta1_pow,
            beta2_pow,
            master_weight,
            None,  # found_inf
            beta1,
            beta2,
            epsilon,
            lr_ratio,
            weight_decay,
            with_decay,
            False,  # lazy_mode
            1000,
            find_master,
            False,
            False,  # amsgrad
        )

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

            # Shared NS + scaling closure (captures ns_steps, epsilon, version, ns_coeffs)
            def ortho_fn(m):
                ns_out = Muon._zeropower_via_newtonschulz5(
                    m,
                    steps=ns_steps,
                    eps=epsilon,
                    ns_coeffs=self._ns_coeffs,
                    ns_matmul_dtype=self._ns_matmul_dtype,
                )
                scaled = Muon._scaling_fn(
                    ns_out, version, self._muon_extra_scale_factor
                )
                return scaled

            # Step 3: Newton-Schulz orthogonalisation
            # Use split_concat_func from param_info if provided, otherwise default to whole matrix
            if (
                param_info is not None
                and param_info.split_concat_func is not None
            ):
                # Use slice function defined in model configuration
                orthogonal_update = param_info.split_concat_func(
                    matrix_2d_global, ortho_fn
                )
                if MUON_DEBUG:
                    _global_rank = paddle.distributed.get_rank()
                    if _global_rank == 0:
                        _sf = param_info.split_concat_func
                        _logger.info(
                            f"[Muon] Using split_concat_func: param={param.name}, "
                            f"split_concat_func={_sf.func.__name__}, "
                            f"args={_sf.args}, kwargs={_sf.keywords}"
                        )
            else:
                # Default: whole matrix orthogonalisation
                orthogonal_update = ortho_fn(matrix_2d_global)

            find_master = param.name in self._master_weights
            master_weight = (
                self._master_weights[param.name] if find_master else None
            )

            with_decay = True
            if (
                self._apply_decay_param_fun is not None
                and not self._apply_decay_param_fun(param.name)
            ):
                with_decay = False
            if with_decay and weight_decay > 0:
                if find_master:
                    master_weight.scale_(1.0 - lr * weight_decay)
                else:
                    param.scale_(1.0 - lr * weight_decay)

            final_step = orthogonal_update * lr

            if find_master:
                master_weight.subtract_(final_step)
                paddle.assign(master_weight.astype(param.dtype), param)
            else:
                param.subtract_(final_step.astype(param.dtype))

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

        # apply for zcc
        self._maybe_refuse()

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
        lr_tensor = paddle.to_tensor(lr, dtype=paddle.float32)
        lr_tensor_f64 = paddle.to_tensor(lr, dtype=paddle.float64)
        for param, grad in muon_params:
            self._muon_update(
                param,
                grad,
                lr_tensor,
                self._get_accumulator(self._moment_acc_str, param),
                group.get("momentum", 0.95),
                group.get("ns_steps", 5),
                group.get("nesterov", True),
                group.get("epsilon", 1e-9),
                wd,
                version=group.get("muon_version", 3),
            )

        # --- Pass 2: AdamW updates ---
        for param, grad in adamw_params:
            self._adamw_update(
                param,
                grad,
                lr_tensor_f64,
                self._get_accumulator(self._moment_acc_str, param),
                self._get_accumulator(self._moment2_acc_str, param),
                self._get_accumulator(self._beta1_pow_acc_str, param),
                self._get_accumulator(self._beta2_pow_acc_str, param),
                group.get("adam_beta1", 0.9),
                group.get("adam_beta2", 0.95),
                group.get("epsilon", 1e-9),
                wd,
            )

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
