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

#include <random>
#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename Context, typename T, int64_t Rank>
struct ArgMaxFunctor {
  void operator()(const Context& ctx,
                  const DenseTensor& in,
                  DenseTensor* index_tensor,
                  const int64_t& axis) {
    auto in_eigen = EigenTensor<T, Rank>::From(in, in.dims());
    auto index_eigen = EigenTensor<int, Rank - 1>::From(*index_tensor);
    index_eigen = in_eigen.argmax(axis).template cast<int>();
  }
};

template <typename Context, typename T>
struct GumbleNoiseGenerator;

template <typename Context, typename T>
struct OneHotGenerator;

template <typename T, typename Context>
void GumbelSoftmaxKernel(const Context& ctx,
                         const DenseTensor& x,
                         float temperature,
                         bool hard,
                         int axis,
                         DenseTensor* out) {
  const int rank = x.dims().size();
  axis = CanonicalAxis(axis, rank);
  int axis_dim = x.dims()[axis];

  PADDLE_ENFORCE_GT(temperature,
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The temperature must be greater than 0. But "
                        "received temperature = %f",
                        temperature));

  // allocate memory on device.
  ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  const int size_to_axis = SizeToAxis(axis, x.dims());
  const int size_from_axis = SizeFromAxis(axis, x.dims());
  DenseTensor x_noise_2d, out_2d;
  x_noise_2d.Resize({size_to_axis, size_from_axis});
  out_2d.ShareDataWith(*out).Resize({size_to_axis, size_from_axis});

  // generate gumbel noise and add it to X
  auto* x_noise_data = ctx.template Alloc<T>(&x_noise_2d);
  GumbleNoiseGenerator<Context, T>::Transform(ctx,
                                              x.data<T>(),
                                              x_noise_data,
                                              size_to_axis,
                                              size_from_axis,
                                              temperature);

#ifdef PADDLE_ON_INFERENCE
  paddle::operators::math::SoftmaxFunctor<Context, T, true>()(
      ctx, axis_dim, &x_noise_2d, &out_2d);
#else
  paddle::operators::math::SoftmaxFunctor<Context, T, false>()(
      ctx, axis_dim, &x_noise_2d, &out_2d);
#endif

  if (hard) {
    OneHotGenerator<Context, T>::Transform(ctx, x, out, axis);
  }
}

}  // namespace phi
