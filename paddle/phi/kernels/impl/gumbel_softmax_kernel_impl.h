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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/funcs/softmax_impl.h"

namespace phi {

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
void GumbelSoftmaxKernelHelper(const Context& ctx,
                               const DenseTensor& x,
                               float temperature,
                               bool hard,
                               int axis,
                               DenseTensor* out) {
  const int rank = x.dims().size();
  axis = funcs::CanonicalAxis(axis, rank);
  int axis_dim = x.dims()[axis];

  PADDLE_ENFORCE_GT(temperature,
                    0,
                    phi::errors::InvalidArgument(
                        "The temperature must be greater than 0. But "
                        "received temperature = %f",
                        temperature));

  // allocate memory on device.
  ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(ctx, out, 1.0);
    return;
  }

  const int size_to_axis = funcs::SizeToAxis(axis, x.dims());
  const int size_from_axis = funcs::SizeFromAxis(axis, x.dims());
  DenseTensor x_noise_2d, out_2d(*out);
  x_noise_2d.Resize({size_to_axis, size_from_axis});
  out_2d.Resize({size_to_axis, size_from_axis});

  // generate gumbel noise and add it to X
  auto* x_noise_data = ctx.template Alloc<T>(&x_noise_2d);
  GumbleNoiseGenerator<Context, T>::Transform(ctx,
                                              x.data<T>(),
                                              x_noise_data,
                                              size_to_axis,
                                              size_from_axis,
                                              temperature);
  phi::funcs::SoftmaxFunctor<Context, T>()(ctx, axis_dim, &x_noise_2d, &out_2d);

  if (hard) {
    OneHotGenerator<Context, T>::Transform(ctx, x, out, axis);
  }
}

template <typename T, typename Context>
void GumbelSoftmaxKernel(const Context& ctx,
                         const DenseTensor& x,
                         float temperature,
                         bool hard,
                         int axis,
                         DenseTensor* out) {
  GumbelSoftmaxKernelHelper<T, Context>(ctx, x, temperature, hard, axis, out);
}

}  // namespace phi
