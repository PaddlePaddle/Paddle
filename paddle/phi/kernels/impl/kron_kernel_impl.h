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

#include <algorithm>
#include <vector>

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "thrust/device_vector.h"
#endif

namespace phi {

inline DenseTensor UnsqueezeTo(const DenseTensor& src, int ndims) {
  const phi::DDim& shape = src.dims();
  int rank = shape.size();
  DenseTensor res;
  res.ShareDataWith(src);
  PADDLE_ENFORCE_LE(
      rank,
      ndims,
      errors::InvalidArgument(
          "The input Tensor's rank should be less than or equal to ndims"
          "Received input Tensor's rank = %d, ndims = %d",
          rank,
          ndims));
  if (rank < ndims) {
    std::vector<int64_t> new_dim(ndims, 1);
    for (int i = ndims - rank; i < ndims; i++) {
      new_dim[i] = shape[i - ndims + rank];
    }
    res.Resize(phi::make_ddim(new_dim));
  }
  return res;
}

template <typename T>
struct KronElemFunctor {
  KronElemFunctor(const T* a,
                  const T* b,
                  T* out,
                  const int64_t* shape_b,
                  const int64_t* stride_a,
                  const int64_t* stride_b,
                  const int64_t* stride_out,
                  int ndims)
      : a_(a),
        b_(b),
        out_(out),
        shape_b_(shape_b),
        stride_a_(stride_a),
        stride_b_(stride_b),
        stride_out_(stride_out),
        ndims_(ndims) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    // it computes 1 element in the output
    int64_t index = idx;
    int64_t index_a = 0;
    int64_t index_b = 0;
    for (int i = 0; i < ndims_; i++) {
      auto pos_i = index / stride_out_[i];
      index = index % stride_out_[i];
      auto pos_ai = pos_i / shape_b_[i];
      auto pos_bi = pos_i % shape_b_[i];
      index_a += stride_a_[i] * pos_ai;
      index_b += stride_b_[i] * pos_bi;
    }
    out_[idx] = a_[index_a] * b_[index_b];
  }

 private:
  const T* a_;
  const T* b_;
  T* out_;
  const int64_t* shape_b_;
  const int64_t* stride_a_;
  const int64_t* stride_b_;
  const int64_t* stride_out_;
  const int ndims_;
};

template <typename Context, typename T>
struct KronOpFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out) {
    int ndims = out->dims().size();
    int64_t numel = out->numel();

    const phi::DDim& dim_x = x.dims();
    const phi::DDim& dim_y = y.dims();
    const phi::DDim& dim_out = out->dims();
    const phi::DDim stride_x = phi::stride(dim_x);
    const phi::DDim stride_y = phi::stride(dim_y);
    const phi::DDim stride_out = phi::stride(dim_out);

    const int64_t *p_stride_x = nullptr, *p_stride_y = nullptr,
                  *p_stride_out = nullptr, *p_shape_y = nullptr;
#if defined(__NVCC__) || defined(__HIPCC__)
    thrust::device_vector<int64_t> d_stride_x(ndims);
    thrust::device_vector<int64_t> d_stride_y(ndims);
    thrust::device_vector<int64_t> d_stride_out(ndims);
    thrust::device_vector<int64_t> d_shape_y(ndims);
    thrust::copy(stride_x.Get(), stride_x.Get() + ndims, d_stride_x.begin());
    thrust::copy(stride_y.Get(), stride_y.Get() + ndims, d_stride_y.begin());
    thrust::copy(
        stride_out.Get(), stride_out.Get() + ndims, d_stride_out.begin());
    thrust::copy(dim_y.Get(), dim_y.Get() + ndims, d_shape_y.begin());

    p_stride_x = thrust::raw_pointer_cast(d_stride_x.data());
    p_stride_y = thrust::raw_pointer_cast(d_stride_y.data());
    p_stride_out = thrust::raw_pointer_cast(d_stride_out.data());
    p_shape_y = thrust::raw_pointer_cast(d_shape_y.data());
#else
    p_stride_x = stride_x.Get();
    p_stride_y = stride_y.Get();
    p_stride_out = stride_out.Get();
    p_shape_y = dim_y.Get();
#endif

    funcs::ForRange<Context> for_range(dev_ctx, numel);
    KronElemFunctor<T> functor(x.data<T>(),
                               y.data<T>(),
                               out->data<T>(),
                               p_shape_y,
                               p_stride_x,
                               p_stride_y,
                               p_stride_out,
                               ndims);
    for_range(functor);
  }
};

template <typename T, typename Context>
void KronKernel(const Context& ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                DenseTensor* out) {
  ctx.template Alloc<T>(out);

  int ndims = out->dims().size();
  DenseTensor xx = UnsqueezeTo(x, ndims);
  DenseTensor yy = UnsqueezeTo(y, ndims);

  KronOpFunctor<Context, T> func;
  func(ctx, xx, yy, out);
}

}  // namespace phi
