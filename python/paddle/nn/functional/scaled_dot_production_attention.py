# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

import paddle
from paddle import _C_ops
from paddle.nn.attention.sdpa import (
    SDPBackend,
    _get_backend_priority,
    _get_enabled_backends,
)
from paddle.nn.functional.flash_attention import _math_attention

if TYPE_CHECKING:
    from paddle import Tensor, dtype
    from paddle.base.core import Place

config = {}
debug_sdpa = True


def init_config():
    global config
    config = {
        "flash_attn": {
            "MINIMUM_SM_VERSION": (8, 0),
            "MAXIMUM_SM_VERSION": (12, 1),
            "support_dtypes": (paddle.float16, paddle.bfloat16)
            if paddle.device.is_bf16_supported()
            else (paddle.float16,),
        },
        "mem_efficient_attn": {
            "MINIMUM_SM_VERSION": (5, 0),
            "MAXIMUM_SM_VERSION": (12, 1),
        },
    }


@dataclass
class SDPParams:
    query_shape: paddle.Size
    key_shape: paddle.Size
    value_shape: paddle.Size
    attn_mask_shape: paddle.Size | None
    attn_strides: list[int] | None
    dropout: float
    is_causal: bool
    scale: float | None
    query_stop_gradient: bool
    dtype: tuple[dtype, dtype, dtype]
    strides: tuple[list[int], list[int], list[int]]
    place: tuple[Place, Place, Place]

    @cached_property
    def batch_size(self) -> tuple[int, int, int]:
        return self.query_shape[0], self.key_shape[0], self.value_shape[0]

    @cached_property
    def seq_len(self) -> tuple[int, int, int]:
        return self.query_shape[1], self.key_shape[1], self.value_shape[1]

    @cached_property
    def num_heads(self) -> tuple[int, int, int]:
        return self.query_shape[2], self.key_shape[2], self.value_shape[2]

    @cached_property
    def head_dim(self) -> tuple[int, int, int]:
        return self.query_shape[-1], self.key_shape[-1], self.value_shape[-1]

    @cached_property
    def device_id(self) -> tuple[int, ...]:
        ret = tuple(
            pl.gpu_device_id() if pl.is_gpu_place() else -1 for pl in self.place
        )
        return ret


@lru_cache(maxsize=8)
def get_device_capability(device_id: int) -> tuple[int, int]:
    if device_id < 0:
        return (0, 0)
    return paddle.device.cuda.get_device_capability(device_id)


def check_sm_version(
    min_sm: tuple[int, int], max_sm: tuple[int, int], device_id: int = 0
) -> bool:
    major, minor = get_device_capability(device_id)
    current = (major, minor)
    return min_sm <= current <= max_sm


@lru_cache(maxsize=1)
def check_cuda_is_available() -> bool:
    return paddle.is_compiled_with_cuda() and paddle.device.is_available()


def check_all_tensors_on_device(params: SDPParams, debug: bool):
    """
    Check all input tensors are placed on the GPU device.
    """
    if not params.place[0].is_gpu_place():
        if debug:
            warnings.warn(
                "All input tensors should be placed on GPU place, but "
                f"query place: {params.place[0]}, key place: "
                f"{params.place[1]}, value place: {params.place[2]}"
            )
        return False
    return True


def check_tensor_shapes(params: SDPParams, debug: bool):
    """
    Check the number of dimensions of QKV are all 4.
    """
    query_dim = len(params.query_shape)
    key_dim = len(params.key_shape)
    value_dim = len(params.value_shape)

    if query_dim != 4 or key_dim != 4 or value_dim != 4:
        if debug:
            warnings.warn(
                "The number of dimensions of query, key, and value should be 4, "
                f"but query_dim: {query_dim}, key_dim: {key_dim}, value_dim: {value_dim}"
            )
        return False
    return True


def check_for_attn_mask(params: SDPParams, debug: bool):
    """
    Check flash attention does not support attn_mask.
    """
    if params.attn_mask_shape is not None:
        if debug:
            warnings.warn("Flash attention does not support attn_mask.")
            return False
    return True


def check_head_dim_size_flash(params: SDPParams, debug: bool):
    """
    Check the dimension of head in query, key, and value should be equal and all less than 256.
    """
    q_head_dim, k_head_dim, v_head_dim = params.head_dim

    if q_head_dim > 256 or q_head_dim != k_head_dim or k_head_dim != v_head_dim:
        if debug:
            warnings.warn(
                "The dimension of head in query, key, and value should be equal and all less than 256, "
                f"but q_head_dim: {q_head_dim}, k_head_dim: {k_head_dim}, v_head_dim: {v_head_dim}"
            )
        return False
    return True


def check_flash_attention_hardware_support(params: SDPParams, debug: bool):
    """
    Check flash attention requires CUDA support and SM between 8.0 and 12.1.
    """
    if not check_cuda_is_available():
        if debug:
            warnings.warn("Flash attention requires CUDA support.")
            return False

    if not check_sm_version(
        config["flash_attn"]["MINIMUM_SM_VERSION"],
        config["flash_attn"]["MAXIMUM_SM_VERSION"],
        params.device_id[0],
    ):
        if debug:
            warnings.warn(
                f"Flash attention requires SM between {config['flash_attn']['MINIMUM_SM_VERSION']}"
                f"and {config['flash_attn']['MAXIMUM_SM_VERSION']}, but found SM "
                f"{get_device_capability(params.device_id)}"
            )
            return False
    return True


def check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120(
    params: SDPParams, debug: bool
):
    if params.query_stop_gradient:
        return True

    maj, min = get_device_capability(params.device_id[0])
    is_consumer_ampere_ada = maj == 8 and min in [6, 9]
    is_blackwell = maj == 12 and min in [0, 1]

    if not (is_consumer_ampere_ada or is_blackwell):
        return True

    hdim = params.head_dim[0]
    dropout = params.dropout

    cond1 = 192 < hdim <= 224
    cond2 = (hdim > 224) and (dropout > 0.0)

    if cond1 or cond2:
        if debug:
            warnings.warn(
                f"Flash Attention training disabled on SM{maj}.{min} "
                f"for head_dim={hdim} and dropout={dropout}. "
                "Constraints: (192, 224] OR (>224 with dropout)."
            )
        return False

    return True


def check_flash_causal_non_square_seqlens(params: SDPParams, debug: bool):
    """
    Check flash attention only supports causal attention when the sequence length of query and key are equal.
    """
    if not params.is_causal:
        return True

    q_len, k_len, _ = params.seq_len
    if q_len == k_len:
        return True

    if debug:
        warnings.warn(
            f"Flash attention only supports causal attention when the sequence"
            f"length of query and key are equal, but got query shape: "
            f"{params.query_shape}, key shape: {params.key_shape}"
        )
    return False


def check_dtypes_low_precision(params: SDPParams, debug: bool):
    """
    check QKV share the same dtype and are supported dtype.
    """
    q_dtype, k_dtype, v_dtype = params.dtype
    if (
        q_dtype != k_dtype
        or v_dtype != k_dtype
        or q_dtype not in config["flash_attn"]["support_dtypes"]
    ):
        if debug:
            warnings.warn(
                f"Flash attention requires query, key, and value "
                f"to be of the same dtype and support dtype, but "
                f"got query dtype: {q_dtype}, key dtype: {k_dtype}"
                f", value dtype: {v_dtype}. Supported dtypes are: "
                f"{config['flash_attn']['support_dtypes']}"
            )
        return False
    return True


def check_batch_size_and_num_heads_dense(
    params: SDPParams, debug: bool = False
):
    """
    Check the batch size and number of heads of query, key, and value are equal.
    """
    # it assumes that dim of kqv is 4, layout is [bs, seq_len, num_heads, head_dim]
    q_bs, k_bs, v_bs = params.batch_size
    q_num_heads, k_num_heads, v_num_heads = params.num_heads

    # our flash attn does not support GQA, so they must equal
    if (
        q_bs != k_bs
        or q_bs != v_bs
        or q_num_heads != k_num_heads
        or q_num_heads != v_num_heads
    ):
        if debug:
            warnings.warn(
                f"Flash attention requires the batch size and number of heads"
                f"of query, key, and value to be equal, but got query shape: "
                f"{params.query_shape}, key shape: {params.key_shape}, value "
                f"shape: {params.value_shape}"
            )
        return False
    return True


def check_nonzero_sequence_lengths_dense(
    params: SDPParams, debug: bool = False
):
    """
    Check the sequence lengths of query and key are non-zero.
    """
    query_seq_len, key_seq_len, _ = params.seq_len
    if query_seq_len == 0 or key_seq_len == 0:
        if debug:
            warnings.warn(
                f"Flash attention requires non-zero sequence lengths, "
                f"but got query shape: {params.query_shape}, key shape: {params.key_shape}"
            )
        return False
    return True


def check_last_dim_stride_equals_1_dense(
    params: SDPParams, debug: bool = False
):
    """
    Check the last dimension stride equals 1.
    """

    if params.query_shape[-1] != 1 and (
        params.strides[0][-1] != 1
        or params.strides[1][-1] != 1
        or params.strides[2][-1] != 1
    ):
        if debug:
            warnings.warn(
                f"Flash attention requires last dimension stride equals 1, "
                f"but got query strides: {params.strides[0]}, key strides:"
                f"{params.strides[1]}, value strides: {params.strides[2]}"
            )
        return False
    if params.attn_strides is not None and params.attn_strides[-1] != 1:
        if debug:
            warnings.warn(
                f"Flash attention requires last dimension stride equals 1, "
                f"but got attn_mask strides: {params.attn_strides}"
            )
        return False
    return True


@lru_cache(maxsize=2)
def use_tensor_cores(is_half: bool, device_id: int) -> bool:
    major, _ = get_device_capability(device_id)
    if major >= 8:
        return True
    if major == 7:
        return is_half
    return False


@lru_cache(maxsize=8)
def minimum_gemm_alignment(dtype: dtype, device_id: int):
    is_half = dtype in (paddle.float16, paddle.bfloat16)
    use_tc = use_tensor_cores(is_half, device_id)
    major, _ = get_device_capability(device_id)
    matmul_alignment_mn = 4 if major > 8 else 1
    bits_per_scalar = 16 if is_half else 32
    if use_tc:
        matmul_alignment_mn = max(matmul_alignment_mn, 128 / bits_per_scalar)
    return matmul_alignment_mn


def check_mem_efficient_hardware_support(params: SDPParams, debug: bool):
    """
    Check mem_efficient attention requires CUDA support and SM between 5.0 and 12.1.
    """
    if not check_cuda_is_available():
        if debug:
            warnings.warn("Mem efficient attention requires CUDA support.")
        return False

    if not check_sm_version(
        config["mem_efficient_attn"]["MINIMUM_SM_VERSION"],
        config["mem_efficient_attn"]["MAXIMUM_SM_VERSION"],
        params.device_id[0],
    ):
        if debug:
            warnings.warn(
                f"Flash attention requires SM between {config['mem_efficient_attn']['MINIMUM_SM_VERSION']}"
                f"and {config['mem_efficient_attn']['MAXIMUM_SM_VERSION']}, but found SM "
                f"{get_device_capability(params.device_id)}"
            )
            return False
    return True


def check_head_dim_size_mem_efficient(params: SDPParams, debug: bool):
    q_head_dim, k_head_dim, v_head_dim = (
        params.query_shape[-1],
        params.key_shape[-1],
        params.value_shape[-1],
    )
    alignment = minimum_gemm_alignment(params.dtype[0], params.device_id[0])
    if (
        q_head_dim % alignment != 0
        or k_head_dim % alignment != 0
        or v_head_dim % alignment != 0
    ):
        if debug:
            warnings.warn(
                f"Mem efficient attention requires head dim size aligned to {alignment}, "
                f"but found q_head_dim: {q_head_dim}, k_head_dim: {k_head_dim}, v_head_dim: {v_head_dim}"
            )
        return False
    return True


def check_attn_mask_alignment(params: SDPParams, debug: bool) -> bool:
    if params.is_causal:
        return True

    if params.attn_mask_shape is None:
        return True

    last_dim = params.attn_mask_shape[-1]

    if last_dim % 8 != 0:
        if debug:
            import warnings

            warnings.warn(
                f"Mem efficient attention requires attn_mask last dimension to be divisible by 8 "
                f"to satisfy vector alignment, but got {last_dim}. "
                "Falling back to other backends."
            )
        return False

    return True


def check_scale_is_None(params: SDPParams, debug: bool) -> bool:
    if params.scale is None:
        return True
    if debug:
        warnings.warn("Paddle's FAV2 does not support scale parameter.")
    return False


def can_use_flash_attention(params: SDPParams, debug: bool = False) -> bool:
    general_constraints = [
        check_all_tensors_on_device,
        check_tensor_shapes,
        check_for_attn_mask,
        check_head_dim_size_flash,
        check_flash_attention_hardware_support,
        check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120,
        check_flash_causal_non_square_seqlens,
        check_dtypes_low_precision,
    ]

    dense_tensor_constraints = [
        check_batch_size_and_num_heads_dense,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense,
    ]

    for constraint in general_constraints:
        if not constraint(params, debug):
            return False

    for constraint in dense_tensor_constraints:
        if not constraint(params, debug):
            return False

    if not check_scale_is_None(params, debug):
        return False
    return True


def can_use_mem_efficient_attention(
    params: SDPParams, debug: bool = False
) -> bool:
    constraints = [
        check_all_tensors_on_device,
        check_mem_efficient_hardware_support,
        check_tensor_shapes,
        check_head_dim_size_mem_efficient,
        check_attn_mask_alignment,
    ]
    for constraint in constraints:
        if not constraint(params, debug):
            return False
    return True


def select_sdp_for_sdpa(param: SDPParams, debug: bool) -> str:
    place = paddle.get_device()
    if "xpu" in place:
        return "flash_attn"

    if "iluvatar_gpu" in place:
        return "flash_attn"

    if "metax_gpu" in place:
        return "flash_attn"

    enabled_backends = _get_enabled_backends()
    priority_order = _get_backend_priority()

    for backend in priority_order:
        if backend not in enabled_backends:
            continue

        if backend == SDPBackend.FLASH_ATTENTION:
            if can_use_flash_attention(param, debug):
                return "flash_attn"
        elif backend == SDPBackend.EFFICIENT_ATTENTION:
            if can_use_mem_efficient_attention(param, debug):
                return "mem_efficient"
        elif backend == SDPBackend.MATH:
            return "math"

    raise RuntimeError(
        "No available backend for scaled_dot_product_attention was found."
    )


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
    name: str | None = None,
    backend: str | None = None,
    scale: float | None = None,
) -> Tensor:
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Note:
        This API differs from :ref:`api_paddle_compat_nn_functional_scaled_dot_product_attention` in that:
            1. The QKV layout of this API is [batch_size, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim].
            2. This API supports GQA(Generic Query Attention) mode.
        If you need GQA mode or num_head first layout, please use ``paddle.compat.nn.functional.scaled_dot_product_attention``.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        3-D tensor with shape:
                        [seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        3-D tensor with shape:
                        [seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        3-D tensor with shape:
                        [seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        attn_mask(Tensor, optional): A float mask of the same type as query,
                        key, value that is added to the attention score.
        dropout_p(float, optional): The dropout ratio.
        is_causal(bool, optional): Whether enable causal mode.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.
        backend(str, optional): Specify which backend to compute scaled dot product attention.
                        Currently only support "p2p" for distribution usage.
        scale(float, optional): The scaling factor used in the calculation of attention weights.
                        If None, scale = 1 / sqrt(head_dim).

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    3-D tensor with shape: [seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('bfloat need V100 compile')
            >>> import paddle
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> output = paddle.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """
    query_ndim = query.ndim
    if query.ndim == 3:
        query = paddle.unsqueeze(query, axis=0)

    if key.ndim == 3:
        key = paddle.unsqueeze(key, axis=0)

    if value.ndim == 3:
        value = paddle.unsqueeze(value, axis=0)

    if (
        backend == 'p2p'
        and query.is_dist()
        and key.is_dist()
        and value.is_dist()
    ):
        # ring attention for auto_parallel mode
        assert scale is None, f"Backend {backend} not support scale parameter."
        out = paddle.distributed.auto_parallel.ring_attention.RingFlashAttention.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
        )
        return out

    param = SDPParams(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        attn_mask_shape=attn_mask.shape if attn_mask is not None else None,
        attn_strides=attn_mask.stride() if attn_mask is not None else None,
        dropout=dropout_p,
        is_causal=is_causal,
        scale=scale,
        query_stop_gradient=query.stop_gradient,
        strides=(query.stride(), key.stride(), value.stride()),
        dtype=(query.dtype, key.dtype, value.dtype),
        place=(query.place, key.place, value.place),
    )
    if len(config) == 0:
        init_config()

    if attn_mask is not None and attn_mask.dtype == paddle.bool:
        attn_mask = paddle.where(
            attn_mask,
            paddle.to_tensor(0.0, dtype=query.dtype),
            paddle.to_tensor(-float('inf'), dtype=query.dtype),
        )
    sdp_func_name = select_sdp_for_sdpa(
        param,
        debug=debug_sdpa,
    )
    print("Selected backend", sdp_func_name)
    if sdp_func_name == "flash_attn":
        fixed_seed_offset = None
        return_softmax = False
        rng_name = ""
        out, _, _, _ = _C_ops.flash_attn(
            query,
            key,
            value,
            fixed_seed_offset,
            attn_mask,
            dropout_p,
            is_causal,
            return_softmax,
            not training,
            rng_name,
        )
    elif sdp_func_name == "mem_efficient":
        from paddle.incubate.nn.memory_efficient_attention import (
            LowerTriangularMask,
            memory_efficient_attention,
        )

        if is_causal:
            bias_input = LowerTriangularMask()
        elif attn_mask is not None:
            bias_input = attn_mask
        else:
            bias_input = None
        if isinstance(bias_input, paddle.Tensor) and bias_input.ndim == 4:
            num_heads = query.shape[2]

            if bias_input.shape[1] == 1 and num_heads > 1:
                target_shape = list(bias_input.shape)
                target_shape[1] = num_heads
                bias_input = bias_input.expand(target_shape)
        out = memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=bias_input,
            p=dropout_p,
            scale=scale,
            training=training,
        )

    elif sdp_func_name == "math":
        out = _math_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            False,
            training,
            scale,
        )[0]
    else:
        raise ValueError(f"Invalid backend {backend}")

    if query_ndim == 3:
        out = paddle.squeeze(out, axis=0)
    return out
