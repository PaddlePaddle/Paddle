// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "paddle/phi/kernels/apply_per_channel_scale_kernel.h"

#include <assert.h>
#include <stdint.h>
#include <cmath>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

namespace {
#ifdef PADDLE_WITH_CUDA
template <typename T>
struct CUDA_HALF_2_TYPE_TARIS {};

template <>
struct CUDA_HALF_2_TYPE_TARIS<half> {
  using type = half2;
};

#ifdef PADDLE_CUDA_BF16
template <>
struct CUDA_HALF_2_TYPE_TARIS<__nv_bfloat16> {
  using type = __nv_bfloat162;
};
#endif

template <typename T>
struct HalfMul2 {};

template <>
struct HalfMul2<half2> {
  static __device__ __forceinline__ half2 apply(const half2& x,
                                                const half2& y) {
    return __hmul2(x, y);
  }
};

#ifdef PADDLE_CUDA_BF16
template <>
struct HalfMul2<__nv_bfloat162> {
  static __device__ __forceinline__ __nv_bfloat162
  apply(const __nv_bfloat162& x, const __nv_bfloat162& y) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmul2(x, y);
#else
    float fxl, fxh, fyl, fyh;
    fxl = __low2float(x);
    fxh = __high2float(x);
    fyl = __low2float(y);
    fyh = __high2float(y);
    return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#endif
  }
};
#endif

template <typename T, int kProcessRows, typename AccessType>
__global__ void apply_per_channel_scale(
    const T* act, const T* scales, int rows, int cols, T* out) {
  using HALF_2_TYPE = typename CUDA_HALF_2_TYPE_TARIS<T>::type;
  static constexpr int kElems = sizeof(AccessType) / sizeof(T);
  T scale[kElems], act_vec[kElems];
  int col_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int row_offset = blockIdx.y;
  if (col_offset * kElems >= cols || row_offset * kProcessRows >= rows) return;
  act += row_offset * kProcessRows * cols;
  out += row_offset * kProcessRows * cols;
  *reinterpret_cast<AccessType*>(scale) =
      reinterpret_cast<const AccessType*>(scales)[col_offset];
#pragma unroll
  for (int i = 0; i < kProcessRows; ++i) {
    *reinterpret_cast<AccessType*>(act_vec) =
        reinterpret_cast<const AccessType*>(act + i * cols)[col_offset];
    if constexpr (kElems % 2 == 0 && (std::is_same<T, half>::value
#ifdef PADDLE_CUDA_BF16
                                      || std::is_same<T, __nv_bfloat16>::value
#endif
                                      )) {
#pragma unroll
      for (int j = 0; j < kElems; j += 2) {
        *reinterpret_cast<HALF_2_TYPE*>(act_vec + j) =
            HalfMul2<HALF_2_TYPE>::apply(
                *reinterpret_cast<HALF_2_TYPE*>(act_vec + j),
                *reinterpret_cast<HALF_2_TYPE*>(scale + j));
      }
    } else {
#pragma unroll
      for (int j = 0; j < kElems; ++j) {
        act_vec[j] *= scale[j];
      }
    }
    reinterpret_cast<AccessType*>(out + i * cols)[col_offset] =
        *reinterpret_cast<AccessType*>(act_vec);
  }
}

template <typename T, int kProcessRows, typename AccessType = float4>
void apply_per_channel_scale_launcher(const T* act,
                                      const T* scales,
                                      int rows,
                                      int cols,
                                      T* out,
                                      cudaStream_t stream = 0) {
  static constexpr int kElems = sizeof(AccessType) / sizeof(T);
  dim3 block(128);
  dim3 grid((cols / kElems + block.x - 1) / block.x,
            (rows + kProcessRows - 1) / kProcessRows);
  apply_per_channel_scale<T, kProcessRows, AccessType>
      <<<grid, block, 0, stream>>>(act, scales, rows, cols, out);
}

}  // namespace
#endif

template <typename T, typename Context>
void ApplyPerChannelScaleKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& scales,
                                DenseTensor* out) {
#ifdef PADDLE_WITH_CUDA
  using DataType = typename PDDataTypeTraits<T>::DataType;
  int rows = x.dims()[0];
  int cols = x.dims()[1];
  int elems = rows * cols;
  const T* x_data = x.data<T>();
  const T* scales_data = scales.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (elems < 2048 * 2048) {
    apply_per_channel_scale_launcher<DataType, 1, float4>(
        reinterpret_cast<const DataType*>(x_data),
        reinterpret_cast<const DataType*>(scales_data),
        rows,
        cols,
        reinterpret_cast<DataType*>(out_data),
        dev_ctx.stream());
  } else if (elems < 4096 * 4096) {
    apply_per_channel_scale_launcher<DataType, 4, float4>(
        reinterpret_cast<const DataType*>(x_data),
        reinterpret_cast<const DataType*>(scales_data),
        rows,
        cols,
        reinterpret_cast<DataType*>(out_data),
        dev_ctx.stream());
  } else if (elems < 8192 * 8192) {
    apply_per_channel_scale_launcher<DataType, 8, float4>(
        reinterpret_cast<const DataType*>(x_data),
        reinterpret_cast<const DataType*>(scales_data),
        rows,
        cols,
        reinterpret_cast<DataType*>(out_data),
        dev_ctx.stream());
  } else {
    apply_per_channel_scale_launcher<DataType, 16, float4>(
        reinterpret_cast<const DataType*>(x_data),
        reinterpret_cast<const DataType*>(scales_data),
        rows,
        cols,
        reinterpret_cast<DataType*>(out_data),
        dev_ctx.stream());
  }
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(apply_per_channel_scale,
                   GPU,
                   ALL_LAYOUT,
                   phi::ApplyPerChannelScaleKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
