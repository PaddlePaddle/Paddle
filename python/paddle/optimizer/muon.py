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

import paddle
from paddle.base import framework
from paddle.distributed.fleet.utils.muon_comm_utils import (
    gather_varlen,
    should_use_muon,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    ShardedStateDict,
    ShardedWeight,
    create_sharded_weight_with_new_local,
)

from ..nn.clip import GradientClipBase
from .optimizer import Optimizer

__all__ = []


class Muon(Optimizer):
    r"""
    Muon optimizer with Sharding/TP-aware parameter updates.

    For 2-D weight matrices (identified by :func:`should_use_muon`), Muon
    applies orthogonal gradient updates via Newton-Schulz iteration.  For all
    other parameters (embeddings, biases, expert weights, …) it falls back to
    a standard AdamW update.

    The optimizer is designed for ``DygraphShardingOptimizerV2``-wrapped usage
    where each parameter is stored as a 1-D shard.  During each step it
    temporarily gathers the full matrix across sharding/TP ranks, performs the
    orthogonal update, then scatters back only the local shard.

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
        is_split_qkv (bool): Apply per-head orthogonalisation for QKV weights.
            Default: ``True``.
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
        nesterov=True,
        adam_epsilon=1e-9,
        grad_clip=None,
        apply_decay_param_fun=None,
        muon_version=1,
        is_split_qkv=True,
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
            "is_split_qkv": is_split_qkv,
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
        self._default_dict.update(defaults)

    # ------------------------------------------------------------------
    # Accumulator management
    # ------------------------------------------------------------------

    def _ensure_accumulators(self, param, use_muon, group):
        """Create optimizer accumulators for *param* if they do not exist yet.

        ``param`` is the 1-D shard held by this rank under ShardingV2.  All
        accumulators share that 1-D shape so that no extra memory is needed.
        """
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
        """Standard entry-point used by checkpoint-resume infrastructure.

        Delegates to _ensure_accumulators so that accumulators are keyed by
        the slice_param name (consistent with _apply_optimize) rather than by
        a master-weight name, which would cause a key mismatch under AMP O2.
        """
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)
        for p in parameters:
            use_muon = should_use_muon(
                p.name, getattr(p, "original_shape", p.shape)
            )
            self._ensure_accumulators(p, use_muon, self._default_dict)

    # ------------------------------------------------------------------
    # Newton-Schulz orthogonalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _zeropower_via_newtonschulz5(X, steps=5, eps=1e-9):
        """Approximate the matrix sign function via 5th-order Newton-Schulz."""
        a, b, c = 3.4445, -4.7750, 2.0315

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

        for _ in range(steps):
            A = paddle.matmul(X, X, transpose_y=True)
            B = paddle.addmm(input=A, x=A, y=A, beta=b, alpha=c)
            X = paddle.addmm(input=X, x=B, y=X, beta=a, alpha=1.0)

        return X.T if transpose else X

    @staticmethod
    def _scaling_fn(orthogonal_update, version):
        """Apply dimension-dependent scaling to the orthogonal update."""
        din, dout = orthogonal_update.shape[0], orthogonal_update.shape[1]
        if version == 1:
            scale = max(1, dout / din) ** 0.5
        elif version == 2:
            scale = (dout / din) ** 0.5
        else:  # version == 3 (default)
            scale = 0.2 * (max(dout, din) ** 0.5)
        return orthogonal_update * scale

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

    @staticmethod
    def _muon_update(
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
        is_split_qkv,
    ):
        """In-place Muon update for 1-D sharded parameters.

        Temporarily gathers the full 2-D weight matrix across sharding and TP
        ranks, applies Newton-Schulz orthogonalisation, then scatters the
        local shard back.
        """
        from paddle.distributed import fleet

        hcg = fleet.get_hybrid_communicate_group()
        sharding_group = hcg.get_sharding_parallel_group()
        tp_group = hcg.get_model_parallel_group()

        is_sharded_gather = getattr(param, "is_sharded_gather", False)
        tp_slice_shape = getattr(param, "original_shape", param.shape)
        split_axis = getattr(param, "split_axis", None)
        sharding_indices = getattr(param, "sharding_indices", None)
        needs_qkv_split = getattr(param, "needs_qkv_split", False)
        is_sharding_v2 = param.ndim == 1

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

            # Step 2: Sharding gather → full TP-slice
            if not is_sharding_v2:
                matrix_2d_tp = update_buffer
            else:
                if is_sharded_gather:
                    assert sharding_indices is not None, (
                        "sharding_indices must be set when is_sharded_gather=True"
                    )
                    s_rank = sharding_group.rank
                    all_shape_and_dtype = [
                        ([length], update_buffer.dtype)
                        if length > 0
                        else (None, None)
                        for length in sharding_indices
                    ]
                    input_tensor = (
                        update_buffer if sharding_indices[s_rank] > 0 else None
                    )

                    for r in range(sharding_group.nranks):
                        if sharding_indices[r] == 0:
                            continue
                        gathered_chunk = gather_varlen(
                            input=input_tensor,
                            dst=sharding_group.ranks[r],
                            group=sharding_group,
                            all_shape_and_dtype=all_shape_and_dtype,
                        )
                        if r == s_rank:
                            full_1d_tp_slice = gathered_chunk

                    if full_1d_tp_slice is None:
                        raise RuntimeError(
                            f"Rank {s_rank} failed to gather Muon param {param.name}"
                        )
                else:
                    full_1d_tp_slice = update_buffer

                tp_numel = 1
                for s in tp_slice_shape:
                    tp_numel *= s
                sharding_total_len = full_1d_tp_slice.numel()
                if sharding_total_len != tp_numel:
                    full_1d_tp_slice = full_1d_tp_slice[:tp_numel]

                matrix_2d_tp = full_1d_tp_slice.reshape(tp_slice_shape)

            # Step 3: TP all-gather → full global matrix
            has_tp = tp_group.nranks > 1
            if has_tp:
                tp_tensor_list = []
                paddle.distributed.all_gather(
                    tp_tensor_list, matrix_2d_tp, group=tp_group
                )
                axis = split_axis if split_axis is not None else -1
                matrix_2d_global = paddle.concat(tp_tensor_list, axis=axis)
            else:
                matrix_2d_global = matrix_2d_tp

            # Step 4: Newton-Schulz orthogonalisation
            if matrix_2d_global.ndim == 3:
                # 3D fused MoE expert tensor [n_experts, H, I].
                # Apply Newton-Schulz independently to each expert's 2D slice.
                n_experts = matrix_2d_global.shape[0]
                expert_updates = []
                for ei in range(n_experts):
                    expert_slice = matrix_2d_global[ei]  # [H, I]
                    expert_ortho = Muon._scaling_fn(
                        Muon._zeropower_via_newtonschulz5(
                            expert_slice, steps=ns_steps, eps=epsilon
                        ),
                        version,
                    )
                    expert_updates.append(expert_ortho)
                orthogonal_update = paddle.stack(expert_updates, axis=0)
            elif is_split_qkv and needs_qkv_split:
                # Per-head update: orthogonalise each Q/K/V head independently.
                # param.head_num / kv_head_num are set by the trainer before
                # the optimizer is constructed.
                head_num = param.head_num
                kv_head_num = param.kv_head_num
                unit_size = matrix_2d_global.shape[1] // (
                    head_num + 2 * kv_head_num
                )
                q_size = head_num * unit_size
                kv_size = kv_head_num * unit_size

                q_block, k_block, v_block = paddle.split(
                    matrix_2d_global, [q_size, kv_size, kv_size], axis=1
                )

                def _ortho_per_head(block, num_heads):
                    """Orthogonalise each head slice independently, then concat."""
                    heads = paddle.split(block, num_heads, axis=1)
                    return paddle.concat(
                        [
                            Muon._scaling_fn(
                                Muon._zeropower_via_newtonschulz5(
                                    h, steps=ns_steps, eps=epsilon
                                ),
                                version,
                            )
                            for h in heads
                        ],
                        axis=1,
                    )

                orthogonal_update = paddle.concat(
                    [
                        _ortho_per_head(q_block, head_num),
                        _ortho_per_head(k_block, kv_head_num),
                        _ortho_per_head(v_block, kv_head_num),
                    ],
                    axis=1,
                )
            else:
                orthogonal_update = Muon._scaling_fn(
                    Muon._zeropower_via_newtonschulz5(
                        matrix_2d_global, steps=ns_steps, eps=epsilon
                    ),
                    version,
                )

            # Step 5: TP scatter → local TP-slice
            if has_tp:
                axis = split_axis if split_axis is not None else -1
                tp_rank = tp_group.rank
                chunk_size = orthogonal_update.shape[axis] // tp_group.nranks
                start = tp_rank * chunk_size
                matrix_2d_tp_new = paddle.slice(
                    orthogonal_update,
                    axes=[axis],
                    starts=[start],
                    ends=[start + chunk_size],
                )
            else:
                matrix_2d_tp_new = orthogonal_update

            # Step 6: Flatten + padding + sharding scatter → local shard
            if is_sharding_v2:
                flat_new = matrix_2d_tp_new.flatten()
                if flat_new.numel() < sharding_total_len:
                    padding = paddle.zeros(
                        [sharding_total_len - flat_new.numel()],
                        dtype=flat_new.dtype,
                    )
                    flat_new = paddle.concat([flat_new, padding])

                if is_sharded_gather:
                    rank = sharding_group.rank
                    start_idx = sum(sharding_indices[:rank])
                    my_len = sharding_indices[rank]
                    local_update_slice = flat_new[
                        start_idx : start_idx + my_len
                    ]
                    if local_update_slice.shape[0] != update_buffer.shape[0]:
                        raise RuntimeError(
                            f"Muon split shape mismatch: got {local_update_slice.shape[0]}, "
                            f"expected {update_buffer.shape[0]} for {param.name}"
                        )
                else:
                    local_update_slice = flat_new
            else:
                local_update_slice = matrix_2d_tp_new

            # Step 7: Apply update with optional weight decay
            if weight_decay > 0:
                param.scale_(1.0 - lr * weight_decay)

            final_step = local_update_slice * lr
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
            use_muon = getattr(
                param,
                "is_muon",
                should_use_muon(
                    param.name, getattr(param, "original_shape", param.shape)
                ),
            )
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
                is_split_qkv=group.get("is_split_qkv", True),
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
                    # V3 MoE: grouped_gemm_experts param.shape is 3D
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
