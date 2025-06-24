// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#ifdef PADDLE_WITH_XPU_FFT
#include <vector>
#include "fft/cuComplex.h"
#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"

namespace xfft_internal::xpu {
int FFTFillConj(int64_t N,
                float2* src_data,
                float2* dst_data,
                const int64_t* src_strides,
                const int64_t* dst_strides,
                const int64_t* dst_shape,
                const bool* is_fft_axis,
                int64_t last_axis,
                int64_t last_axis_size,
                int64_t rank);
int FFTFillConjGrad(int N,
                    float2* input,
                    int64_t axis,
                    int64_t stride_second_to_last_axis,
                    int64_t stride_to_last_axis,
                    size_t double_length);
}  // namespace xfft_internal::xpu

namespace phi {
namespace funcs {
template <typename DeviceContext, typename C>
void FFTFillConj(const DeviceContext& ctx,
                 DenseTensor* src,
                 DenseTensor* dst,
                 const std::vector<int64_t>& axes) {
  std::vector<int64_t> src_strides_v =
      common::vectorize<int64_t>(common::stride(src->dims()));
  std::vector<int64_t> dst_strides_v =
      common::vectorize<int64_t>(common::stride(dst->dims()));
  std::vector<int64_t> dst_shape_v = common::vectorize<int64_t>(dst->dims());
  auto src_data = src->data<C>();
  auto dst_data = dst->data<C>();
  auto last_axis = axes.back();
  auto last_axis_size = dst->dims().at(last_axis) / 2 + 1;
  int64_t rank = dst->dims().size();
  auto _is_fft_axis = std::make_unique<bool[]>(rank);
  for (const auto i : axes) {
    _is_fft_axis[i] = true;
  }
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  int64_t* src_strides_ptr =
      RAII_GUARD.alloc_l3_or_gm<int64_t>(src_strides_v.size());
  PADDLE_ENFORCE_NOT_NULL(src_strides_ptr,
                          common::errors::External("XPU has no enough memory"));
  xpu_memcpy(reinterpret_cast<void*>(src_strides_ptr),
             reinterpret_cast<void*>(src_strides_v.data()),
             src_strides_v.size() * sizeof(int64_t),
             XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  int64_t* dst_strides_ptr =
      RAII_GUARD.alloc_l3_or_gm<int64_t>(dst_strides_v.size());
  PADDLE_ENFORCE_NOT_NULL(dst_strides_ptr,
                          common::errors::External("XPU has no enough memory"));
  xpu_memcpy(reinterpret_cast<void*>(dst_strides_ptr),
             reinterpret_cast<void*>(dst_strides_v.data()),
             dst_strides_v.size() * sizeof(int64_t),
             XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  int64_t* dst_shape_ptr =
      RAII_GUARD.alloc_l3_or_gm<int64_t>(dst_shape_v.size());
  PADDLE_ENFORCE_NOT_NULL(dst_shape_ptr,
                          common::errors::External("XPU has no enough memory"));
  xpu_memcpy(reinterpret_cast<void*>(dst_shape_ptr),
             reinterpret_cast<void*>(dst_shape_v.data()),
             dst_shape_v.size() * sizeof(int64_t),
             XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  bool* _is_fft_axis_ptr = RAII_GUARD.alloc_l3_or_gm<bool>(rank);
  PADDLE_ENFORCE_NOT_NULL(_is_fft_axis_ptr,
                          common::errors::External("XPU has no enough memory"));
  xpu_memcpy(reinterpret_cast<void*>(_is_fft_axis_ptr),
             reinterpret_cast<void*>(_is_fft_axis.get()),
             rank * sizeof(bool),
             XPUMemcpyKind::XPU_HOST_TO_DEVICE);

  int r = xfft_internal::xpu::FFTFillConj(
      dst->numel(),
      reinterpret_cast<cuFloatComplex*>(src_data),
      reinterpret_cast<cuFloatComplex*>(dst_data),
      src_strides_ptr,
      dst_strides_ptr,
      dst_shape_ptr,
      _is_fft_axis_ptr,
      static_cast<int64_t>(last_axis),
      static_cast<int64_t>(last_axis_size),
      static_cast<int64_t>(rank));
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}

template <typename DeviceContext, typename C>
void FFTFillConjGrad(const DeviceContext& ctx,
                     const DenseTensor& out_grad,
                     const std::vector<int64_t>& axes,
                     DenseTensor* x_grad) {
  size_t double_length =
      out_grad.dims()[axes.back()] - x_grad->dims()[axes.back()];
  int64_t stride_to_last_axis = 1;
  auto ddim = x_grad->dims();
  for (int i = ddim.size() - 2; i >= axes.back(); --i) {
    stride_to_last_axis *= ddim[i + 1];
  }
  int64_t stride_second_to_last_axis = stride_to_last_axis * ddim[axes.back()];
  int r = xfft_internal::xpu::FFTFillConjGrad(
      x_grad->numel(),
      reinterpret_cast<cuFloatComplex*>(x_grad->data<C>()),
      axes.back(),
      stride_second_to_last_axis,
      stride_to_last_axis,
      double_length);
  PADDLE_ENFORCE_XPU_SUCCESS(r);
}

}  // namespace funcs
}  // namespace phi
#endif
