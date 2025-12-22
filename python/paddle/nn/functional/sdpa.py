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

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

import paddle
from paddle import _C_ops
from paddle.base.log_helper import get_logger
from paddle.nn.attention.sdpa import (
    SDPBackend,
    _get_backend_priority,
    _get_enabled_backends,
)

_logger = get_logger(
    __name__, "INFO", fmt='%(asctime)s-%(levelname)s: %(message)s'
)
from paddle.nn.functional.flash_attention import _math_attention

if TYPE_CHECKING:
    from paddle import Tensor, dtype
    from paddle.base.core import Place

_config = {}


def init_config():
    global _config
    _config = {
        "flash_attn": {
            "MINIMUM_SM_VERSION": (8, 0),
            "MAXIMUM_SM_VERSION": (12, 1),
            "support_dtypes": (paddle.float16, paddle.bfloat16)
            if paddle.device.is_bf16_supported(including_emulation=False)
            else (paddle.float16,),
        },
        "mem_efficient_attn": {
            "MINIMUM_SM_VERSION": (5, 0),
            "MAXIMUM_SM_VERSION": (12, 1),
            "support_dtypes": (
                paddle.float16,
                paddle.bfloat16,
                paddle.float,
            )
            if paddle.device.is_bf16_supported(including_emulation=False)
            else (paddle.float16, paddle.float),
        },
    }


def _repeat_kv(key: Tensor, value: Tensor, num_repeats: int):
    """
    Repeat key and value tensors along the num_heads(3) dimension. The layout
    of key and value should be [batch_size, seq_len, num_heads, head_dim].
    """
    if num_repeats == 1:
        return key, value
    # repeat_interleave does not support float16 on GPU, so we manually expand the tensor
    key, value = key.unsqueeze(3), value.unsqueeze(3)
    key, value = (
        key.expand([-1, -1, -1, num_repeats, -1]),
        value.expand([-1, -1, -1, num_repeats, -1]),
    )
    key, value = (
        key.flatten(2, 3).contiguous(),
        value.flatten(2, 3).contiguous(),
    )
    return key, value


@dataclass
class SDPParams:
    query_shape: paddle.Size
    key_shape: paddle.Size
    value_shape: paddle.Size
    attn_mask_shape: paddle.Size | None
    dropout: float
    is_causal: bool
    scale: float | None
    query_stop_gradient: bool
    dtype: tuple[dtype, dtype, dtype]
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


@lru_cache(maxsize=32)
def check_sm_version(
    min_sm: tuple[int, int], max_sm: tuple[int, int], device_id: int = 0
) -> bool:
    major, minor = get_device_capability(device_id)
    current = (major, minor)
    return min_sm <= current <= max_sm


@lru_cache(maxsize=1)
def check_cuda_is_available() -> bool:
    return paddle.is_compiled_with_cuda() and paddle.cuda.is_available()


def check_all_tensors_on_device(params: SDPParams):
    """
    Check all input tensors are placed on the GPU device.
    """
    if not (
        params.place[0].is_gpu_place() or params.place[0].is_custom_place()
    ):
        _logger.debug(
            "All input tensors should be placed on GPU or custom place, but "
            f"query place: {params.place[0]}, key place: "
            f"{params.place[1]}, value place: {params.place[2]}"
        )
        return False
    return True


def check_head_dim_size_flash(params: SDPParams):
    """
    Check the dimension of head in query, key, and value should be equal and all less than 256.
    """
    q_head_dim, k_head_dim, v_head_dim = params.head_dim

    if q_head_dim > 256 or q_head_dim != k_head_dim or k_head_dim != v_head_dim:
        _logger.debug(
            "The dimension of head in query, key, and value should be equal and all less than 256, "
            f"but q_head_dim: {q_head_dim}, k_head_dim: {k_head_dim}, v_head_dim: {v_head_dim}"
        )
        return False
    if q_head_dim % 8 != 0:
        _logger.debug(
            "The dimension of head in query, key, and value should be a multiple of 8, "
            f"but q_head_dim: {q_head_dim}"
        )
        return False
    return True


@lru_cache(maxsize=8)
def check_flash_attention_hardware_support(device_id: int):
    """
    Check flash attention requires CUDA support and SM between 8.0 and 12.1.
    """
    if SDPBackend.FLASH_ATTENTION and paddle.is_compiled_with_custom_device(
        paddle.device.get_all_device_type()[0]
    ):
        return True

    if not check_cuda_is_available():
        _logger.debug("Flash attention requires CUDA support.")
        return False

    if not check_sm_version(
        _config["flash_attn"]["MINIMUM_SM_VERSION"],
        _config["flash_attn"]["MAXIMUM_SM_VERSION"],
        device_id,
    ):
        _logger.debug(
            f"Flash attention requires SM between {_config['flash_attn']['MINIMUM_SM_VERSION']}"
            f"and {_config['flash_attn']['MAXIMUM_SM_VERSION']}, but found SM "
            f"{get_device_capability(device_id)}"
        )
        return False
    return True


def check_flash_causal_non_square_seqlens(params: SDPParams):
    """
    Check flash attention only supports causal attention when the sequence length of query and key are equal.
    """
    if not params.is_causal:
        return True

    q_len, k_len, _ = params.seq_len
    if q_len == k_len:
        return True

    _logger.debug(
        f"Flash attention only supports causal attention when the sequence"
        f"length of query and key are equal, but got query shape: "
        f"{params.query_shape}, key shape: {params.key_shape}"
    )
    return False


def check_dtypes_low_precision_fa(params: SDPParams):
    """
    check QKV share the same dtype and are supported dtype.
    """
    q_dtype, k_dtype, v_dtype = params.dtype
    if (
        q_dtype != k_dtype
        or v_dtype != k_dtype
        or q_dtype not in _config["flash_attn"]["support_dtypes"]
    ):
        _logger.debug(
            f"Flash attention requires query, key, and value "
            f"to be of the same dtype and support dtype, but "
            f"got query dtype: {q_dtype}, key dtype: {k_dtype}"
            f", value dtype: {v_dtype}. Supported dtypes are: "
            f"{_config['flash_attn']['support_dtypes']}"
        )
        return False
    return True


def check_dtypes_low_precision_mem_efficient_attn(params: SDPParams):
    """
    check QKV share the same dtype and are supported dtype.
    """
    q_dtype, k_dtype, v_dtype = params.dtype
    if (
        q_dtype != k_dtype
        or v_dtype != k_dtype
        or q_dtype not in _config["mem_efficient_attn"]["support_dtypes"]
    ):
        _logger.debug(
            f"Mem_efficient_attn requires query, key, and value "
            f"to be of the same dtype and support dtype, but "
            f"got query dtype: {q_dtype}, key dtype: {k_dtype}"
            f", value dtype: {v_dtype}. Supported dtypes are: "
            f"{_config['mem_efficient_attn']['support_dtypes']}"
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


@lru_cache(maxsize=32)
def minimum_gemm_alignment(dtype: dtype, device_id: int):
    is_half = dtype in (paddle.float16, paddle.bfloat16)
    use_tc = use_tensor_cores(is_half, device_id)
    major, _ = get_device_capability(device_id)
    matmul_alignment_mn = 4 if major > 8 else 1
    bits_per_scalar = 16 if is_half else 32
    if use_tc:
        matmul_alignment_mn = max(matmul_alignment_mn, 128 / bits_per_scalar)
    return matmul_alignment_mn


@lru_cache(maxsize=8)
def check_mem_efficient_hardware_support(device_id: int):
    """
    Check mem_efficient attention requires CUDA support and SM between 5.0 and 12.1.
    """
    if not check_cuda_is_available():
        _logger.debug("Mem efficient attention requires CUDA support.")
        return False

    if not check_sm_version(
        _config["mem_efficient_attn"]["MINIMUM_SM_VERSION"],
        _config["mem_efficient_attn"]["MAXIMUM_SM_VERSION"],
        device_id,
    ):
        _logger.debug(
            f"Mem efficient attention requires SM between {_config['mem_efficient_attn']['MINIMUM_SM_VERSION']}"
            f"and {_config['mem_efficient_attn']['MAXIMUM_SM_VERSION']}, but found SM "
            f"{get_device_capability(device_id)}"
        )
        return False
    return True


def check_head_dim_size_mem_efficient(params: SDPParams):
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
        _logger.debug(
            f"Mem efficient attention requires head dim size aligned to {alignment}, "
            f"but found q_head_dim: {q_head_dim}, k_head_dim: {k_head_dim}, v_head_dim: {v_head_dim}"
        )
        return False
    return True


def check_attn_mask_alignment(params: SDPParams) -> bool:
    if params.is_causal:
        return True

    if params.attn_mask_shape is None:
        return True

    last_dim = params.attn_mask_shape[-1]

    if last_dim % 8 != 0:
        _logger.debug(
            f"Mem efficient attention requires attn_mask last dimension to be divisible by 8 "
            f"to satisfy vector alignment, but got {last_dim}. "
            "Falling back to other backends."
        )
        return False

    return True


def check_scale_is_None(params: SDPParams) -> bool:
    if params.scale is None:
        return True
    _logger.debug("Paddle's FAV2 does not support scale parameter.")
    return False


def can_use_flash_attention(params: SDPParams = False) -> bool:
    general_constraints = [
        check_all_tensors_on_device,
        check_head_dim_size_flash,
        check_flash_causal_non_square_seqlens,
        check_dtypes_low_precision_fa,
        check_scale_is_None,
    ]

    for constraint in general_constraints:
        if not constraint(params):
            return False

    if not check_flash_attention_hardware_support(params.device_id[0]):
        return False

    return True


def can_use_mem_efficient_attention(params: SDPParams = False) -> bool:
    constraints = [
        check_all_tensors_on_device,
        check_head_dim_size_mem_efficient,
        check_attn_mask_alignment,
        check_dtypes_low_precision_mem_efficient_attn,
    ]
    for constraint in constraints:
        if not constraint(params):
            return False
    if not check_mem_efficient_hardware_support(params.device_id[0]):
        return False
    return True


def select_sdp_for_sdpa(param: SDPParams) -> str:
    # Note: This API is designed for nn.functional.scaled_dot_product_attention,
    # and is **NOT** expected to be called by others. Some promises should be guaranteed
    # by caller to skip some rarely unmet constraints:
    # 1. The input dim is 4, layout is (batch, seq_len, num_heads, head_dim)
    # 2. The batch_size and num_heads of each input should be the same

    place = paddle.get_device()
    if "xpu" in place:
        return "flash_attn"

    enabled_backends = _get_enabled_backends()
    priority_order = _get_backend_priority()

    for backend in priority_order:
        if backend not in enabled_backends:
            continue

        if backend == SDPBackend.FLASH_ATTENTION:
            if can_use_flash_attention(param):
                return "flash_attn"
        elif backend == SDPBackend.EFFICIENT_ATTENTION:
            if can_use_mem_efficient_attention(param):
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
    backend: str | None = None,
    scale: float | None = None,
    enable_gqa: bool = True,
    name: str | None = None,
) -> Tensor:
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only verifies inputs with dtype float16 and bfloat16, other dtypes may fall back to math
            implementation, which is less optimized.

    Warning:
        If is_causal is set to True, the causal mask should not be provided, otherwise
            the provided mask will be ignored.

    Note:
        This API differs from :ref:`api_paddle_compat_nn_functional_scaled_dot_product_attention` in that:
            1. The QKV layout of this API is [batch_size, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim].
        If you need num_heads before seq_len layout, please use ``paddle.compat.nn.functional.scaled_dot_product_attention``.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len_key, num_heads, head_dim].
                        3-D tensor with shape:
                        [seq_len_key, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len_key, num_heads, head_dim].
                        3-D tensor with shape:
                        [seq_len_key, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len_value, num_heads, head_dim].
                        3-D tensor with shape:
                        [seq_len_value, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        attn_mask(Tensor, optional): The attention mask tensor. The shape should be broadcastable to
                        [batch_size, num_heads, seq_len_key, seq_len_query]. The dtype can be bool
                        or same type of query. The bool mask indicates the positions should take part
                        in attention. The non-bool mask will be added to attention score.
        dropout_p(float, optional): The dropout ratio.
        is_causal(bool, optional): Whether enable causal mode.
        training(bool, optional): Whether it is in the training phase.
        backend(str, optional): Specify which backend to compute scaled dot product attention.
                        Currently only support "p2p" for distribution usage.
        scale(float, optional): The scaling factor used in the calculation of attention weights.
                        If None, scale = 1 / sqrt(head_dim).
        enable_gqa(bool, optional): Whether enable GQA(Group Query Attention) mode. Default is True.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

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
    is_batched = query.dim() == 4
    if not is_batched:
        # FlashAttention backend does not support unbatched input,
        # we add batch dim here and will skip check input dim when selecting FA backend.
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
    k_heads, q_heads, v_heads = (
        key.shape[2],
        query.shape[2],
        value.shape[2],
    )
    if enable_gqa:
        assert k_heads == 0 or q_heads % k_heads == 0, (
            f"The number of groups in query({q_heads}) must be divisible by the number of groups in key({k_heads}) if GQA enabled."
        )
        assert k_heads == v_heads, (
            f"The number of groups in key({k_heads}) must be equal to the number of groups in value({v_heads}) if GQA enabled."
        )
    else:
        assert q_heads == k_heads == v_heads, (
            f"The number of groups in query({q_heads}) must be equal to the number of groups in key({k_heads}) "
            f"and the number of groups in value({v_heads}) if GQA disabled."
        )
    bs, seq_len_q, num_heads_q, head_dim_q = query.shape
    _, seq_len_k, num_heads_k, head_dim_k = key.shape

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

    if not paddle.base.in_dygraph_mode():
        qkv_place = (paddle.framework._current_expected_place_(),) * 3
    else:
        qkv_place = (query.place, key.place, value.place)

    param = SDPParams(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        attn_mask_shape=attn_mask.shape if attn_mask is not None else None,
        dropout=dropout_p,
        is_causal=is_causal,
        scale=scale,
        query_stop_gradient=query.stop_gradient,
        dtype=(query.dtype, key.dtype, value.dtype),
        place=qkv_place,
    )
    if len(_config) == 0:
        init_config()

    is_zero_size = (
        query.shape.numel() == 0
        or key.shape.numel() == 0
        or value.shape.numel() == 0
    )

    if attn_mask is not None:
        if attn_mask.dtype == paddle.bool:
            attn_mask = paddle.where(
                attn_mask,
                paddle.to_tensor(0.0, dtype=query.dtype),
                paddle.to_tensor(-float('inf'), dtype=query.dtype),
            )

    if is_zero_size:
        sdp_func_name = "math"
    else:
        sdp_func_name = select_sdp_for_sdpa(param)

    _logger.debug("Selected backend:" + sdp_func_name)
    if sdp_func_name == "flash_attn":
        fixed_seed_offset = None
        return_softmax = False
        rng_name = ""
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.expand([bs, 1, *attn_mask.shape])
            elif attn_mask.ndim == 3:
                attn_mask = paddle.unsqueeze(attn_mask, axis=1)

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

        repeats = q_heads // k_heads
        key, value = _repeat_kv(key, value, repeats)

        if is_causal:
            attn_mask = LowerTriangularMask()
        elif attn_mask is not None:
            # memory_efficient_attention does not support broadcast num_heads dim when batch_size dim is not 1
            if (
                attn_mask.dim() == 4
                and attn_mask.shape[0] != 1
                and attn_mask.shape[1] != num_heads_q
            ):
                attn_mask = attn_mask.expand(
                    [
                        attn_mask.shape[0],
                        num_heads_q,
                        attn_mask.shape[2],
                        attn_mask.shape[3],
                    ]
                )
        out = memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attn_mask,
            p=dropout_p,
            scale=scale,
            training=training,
        )

    elif sdp_func_name == "math":
        repeats = q_heads // k_heads if k_heads != 0 else 1
        key, value = _repeat_kv(key, value, repeats)
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

    if not is_batched:
        out = paddle.squeeze(out, axis=0)
    return out
