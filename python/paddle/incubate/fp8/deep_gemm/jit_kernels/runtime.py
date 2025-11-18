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

# The file has been adapted from DeepSeek DeepGEMM project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE

from typing import Any

import cuda.bindings.driver as cbd

import paddle

from ..jit.runtime import GemmType
from .utils import get_tma_aligned_size

# TODO Support dtype in Paddle
tmap_type_map: dict[Any, str] = {
    paddle.int8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    paddle.int16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    paddle.int32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    paddle.int64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    paddle.uint8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    # paddle.uint16:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    # paddle.uint32:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    # paddle.uint64:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    paddle.float32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    paddle.float16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    paddle.bfloat16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    paddle.float8_e4m3fn: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    # paddle.float8_e4m3fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    paddle.float8_e5m2: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    # paddle.float8_e5m2fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_type_map = {
    0: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


def make_2d_tma_copy_desc(
    t: paddle.Tensor,
    gmem_dims: tuple[cbd.cuuint64_t, cbd.cuuint64_t],
    gmem_outer_stride: cbd.cuuint64_t,
    smem_dims: tuple[cbd.cuuint32_t, cbd.cuuint32_t],
    swizzle_type: cbd.CUtensorMapSwizzle,
) -> cbd.CUtensorMap:
    tensor_dtype = tmap_type_map[t.dtype]
    res, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        2,
        t.data_ptr(),
        gmem_dims,
        (gmem_outer_stride,),
        smem_dims,
        (cbd.cuuint32_t(1), cbd.cuuint32_t(1)),
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type,
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if res != cbd.CUresult.CUDA_SUCCESS:
        raise Exception(f"Failed to encode tensor map: {res}")
    return tensor_map


def make_2d_tma_desc(
    t: paddle.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    gmem_outer_stride: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    swizzle_type: cbd.CUtensorMapSwizzle = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
) -> cbd.CUtensorMap:
    gmem_dim = (cbd.cuuint64_t(gmem_inner_dim), cbd.cuuint64_t(gmem_outer_dim))
    smem_dim = (cbd.cuuint32_t(smem_inner_dim), cbd.cuuint32_t(smem_outer_dim))
    return make_2d_tma_copy_desc(
        t,
        gmem_dim,
        cbd.cuuint64_t(gmem_outer_stride * t.element_size()),
        smem_dim,
        swizzle_type,
    )


def make_2d_tma_a_desc(
    gemm_type: GemmType,
    t: paddle.Tensor,
    shape_m: int,
    shape_k: int,
    m_stride: int,
    block_m: int,
    block_k: int,
    num_groups: int,
) -> cbd.CUtensorMap:
    return make_2d_tma_desc(
        t,
        shape_k,
        shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1),
        m_stride,
        block_k,
        block_m,
    )


def make_2d_tma_b_desc(
    gemm_type: GemmType,
    t: paddle.Tensor,
    shape_n: int,
    shape_k: int,
    n_stride: int,
    block_n: int,
    block_k: int,
    num_groups: int,
) -> cbd.CUtensorMap:
    return make_2d_tma_desc(
        t,
        shape_k,
        shape_n * (num_groups if gemm_type != GemmType.Normal else 1),
        n_stride,
        block_k,
        block_n,
    )


def make_2d_tma_d_desc(
    gemm_type: GemmType,
    t: paddle.Tensor,
    shape_m: int,
    shape_n: int,
    m_stride: int,
    block_m: int,
    block_n: int,
    num_groups: int,
    swizzle_mode: int,
) -> cbd.CUtensorMap:
    # Swizzling requires the inner box dim to be less or equal than `kSwizzleDMode`
    # bytes, so `BLOCK_N * sizeof(T) / kSwizzleDMode` TMA stores are required
    return make_2d_tma_desc(
        t,
        shape_n,
        shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1),
        m_stride,
        block_n if swizzle_mode == 0 else swizzle_mode // t.element_size(),
        block_m,
        swizzle_type_map[swizzle_mode],
    )


def make_2d_tma_scales_desc(
    gemm_type: GemmType,
    t: paddle.Tensor,
    shape_mn: int,
    shape_k: int,
    block_mn: int,
    block_k: int,
    num_groups: int,
) -> cbd.CUtensorMap:
    # Make TMA aligned to 16 bytes
    shape_mn = get_tma_aligned_size(shape_mn, t.element_size())
    return make_2d_tma_desc(
        t,
        shape_mn,
        (shape_k + block_k - 1)
        // block_k
        * (num_groups if gemm_type == GemmType.GroupedMasked else 1),
        shape_mn,
        block_mn,
        1,
        cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    )
