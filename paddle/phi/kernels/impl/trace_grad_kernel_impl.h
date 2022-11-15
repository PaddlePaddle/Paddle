// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <algorithm>

#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
struct TraceGradFunctor {
  TraceGradFunctor(const T* d_out,
                   const int64_t* out_stride,
                   const int64_t* x_strides,
                   int64_t pos,
                   int64_t dim_size,
                   int64_t dim1,
                   int64_t dim2,
                   int64_t diag_size,
                   T* d_x)
      : d_out_(d_out),
        out_stride_(out_stride),
        x_strides_(x_strides),
        pos_(pos),
        dim_size_(dim_size),
        dim1_(dim1),
        dim2_(dim2),
        diag_size_(diag_size),
        d_x_(d_x) {}

  HOSTDEVICE void operator()(size_t idx) const {
    int64_t num = idx - pos_;
    int64_t position = 0;
    if (num >= 0) {
      int64_t dim1 = 0;
      int64_t dim2 = 0;
      int64_t out_idx = 0;
      for (int64_t i = 0; i < dim_size_; i++) {
        if (i != dim1_ && i != dim2_) {
          position += num / x_strides_[i] * out_stride_[out_idx++];
        } else if (i == dim1_) {
          dim1 = num / x_strides_[i];
        } else {
          dim2 = num / x_strides_[i];
        }
        num = num % x_strides_[i];
      }
      if (dim1 == dim2 && dim1 < diag_size_) {
        d_x_[idx] = d_out_[position];
      }
    }
  }
  const T* d_out_;
  const int64_t* out_stride_;
  const int64_t* x_strides_;
  int64_t pos_;
  int64_t dim_size_;
  int64_t dim1_;
  int64_t dim2_;
  int64_t diag_size_;
  T* d_x_;
};

template <typename T, typename Context>
void TraceGradKernel(const Context& ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     int offset,
                     int axis1,
                     int axis2,
                     DenseTensor* in_grad) {
  auto input_dims = in_grad->dims();
  auto input_stride = phi::stride(input_dims);
  auto output_dims = out_grad.dims();
  auto output_stride = phi::stride(output_dims);

  auto* out_data = out_grad.data<T>();
  T* x_data = ctx.template Alloc<T>(in_grad);

  phi::funcs::SetConstant<Context, T> set_zero;

  set_zero(ctx, in_grad, static_cast<T>(0.0));
  auto dim1 = axis1;
  auto dim2 = axis2;
  auto dim1_ = dim1 < 0 ? input_dims.size() + dim1 : dim1;
  auto dim2_ = dim2 < 0 ? input_dims.size() + dim2 : dim2;
  auto len1 = input_dims[std::min(dim1_, dim2_)];
  auto len2 = input_dims[std::max(dim1_, dim2_)];
  auto stride1 = input_stride[std::min(dim1_, dim2_)];
  auto stride2 = input_stride[std::max(dim1_, dim2_)];

  int offset_stride = 0;
  if (offset >= 0) {
    offset_stride = stride2;
    len2 -= offset;
  } else {
    offset_stride = stride1;
    len1 += offset;
  }
  int64_t diag_size = len2 < len1 ? len2 : len1;
  int64_t pos = std::abs(offset) * offset_stride;
  if (diag_size > 0) {
#if defined(__NVCC__) || defined(__HIPCC__)
    thrust::device_vector<int64_t> output_vec(vectorize(output_stride));
    const int64_t* output_arr = thrust::raw_pointer_cast(output_vec.data());
    thrust::device_vector<int64_t> input_vec(vectorize(input_stride));
    const int64_t* input_arr = thrust::raw_pointer_cast(input_vec.data());

#else
    const auto* output_arr = output_stride.Get();
    const auto* input_arr = input_stride.Get();
#endif

    phi::funcs::ForRange<Context> for_range(ctx, in_grad->numel());
    TraceGradFunctor<T> functor(out_data,
                                output_arr,
                                input_arr,
                                pos,
                                input_dims.size(),
                                dim1_,
                                dim2_,
                                diag_size,
                                x_data);
    for_range(functor);
  }
}

}  // namespace phi
