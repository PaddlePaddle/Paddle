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
#include <type_traits>
#include <vector>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"
#include "paddle/phi/kernels/logsumexp_grad_kernel.h"

#include "paddle/common/flags.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

// Original gradient functor: uses MPType promotion (float16->float32,
// float32->float64) for intermediate computation, then casts back.
template <typename T>
struct LogsumexpGradFunctor {
  template <typename Context,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const Context& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size UNUSED) {
    using MT = typename dtype::MPTypeTrait<T>::Type;
    auto x_mt = (*x).template cast<MT>();
    auto y_mt = (*y).template cast<MT>();
    auto dy_mt = (*dy).template cast<MT>();
    dx->device(place) =
        (dy_mt.broadcast(dim) * (x_mt - y_mt.broadcast(dim)).exp())
            .template cast<T>();
  }
};

// PyTorch-compatible gradient functor: computes in native precision (no MPType
// promotion). Matches PyTorch's logsumexp_backward:
//   grad * (self - result).exp()
// where all arithmetic is in the tensor's native dtype.
template <typename T>
struct LogsumexpGradCompatibleFunctor {
  template <typename Context,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const Context& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size UNUSED) {
    dx->device(place) =
        (*dy).broadcast(dim) * ((*x) - (*y).broadcast(dim)).exp();
  }
};

template <typename T, typename Context, typename Functor>
void LogsumexpGradImpl(const Context& dev_ctx,
                       const DenseTensor& in,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       const std::vector<int64_t>& axis,
                       bool reduce_all,
                       DenseTensor* in_grad) {
  if (reduce_all) {
    auto x = EigenVector<T>::Flatten(in);
    auto y = EigenVector<T>::Flatten(out);
    auto dy = EigenVector<T>::Flatten(out_grad);
    auto dx = EigenVector<T>::Flatten(*in_grad);
    auto& place = *dev_ctx.eigen_device();
    auto broadcast_dim = Eigen::array<int, 1>({{static_cast<int>(in.numel())}});
    Functor()(place, &x, &y, &dx, &dy, broadcast_dim, broadcast_dim[0]);
  } else {
    int rank = in.dims().size();
    Functor functor;
    std::vector<int32_t> axis32;
    axis32.reserve(axis.size());
    std::for_each(axis.begin(), axis.end(), [&axis32](const int64_t& t) {
      axis32.push_back(static_cast<int32_t>(t));
    });
    switch (rank) {
      case 1:
        funcs::ReduceGradFunctor<Context, T, 1, Functor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 2:
        funcs::ReduceGradFunctor<Context, T, 2, Functor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 3:
        funcs::ReduceGradFunctor<Context, T, 3, Functor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 4:
        funcs::ReduceGradFunctor<Context, T, 4, Functor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      default:
        PADDLE_THROW(common::errors::Unimplemented(
            "Unsupported dimensions, please keep maximum dimensions of input "
            "data less than 4."));
        break;
    }
  }
}

template <typename T, typename Context>
void LogsumexpGradKernel(const Context& dev_ctx,
                         const DenseTensor& in,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis_in,
                         bool keepdim UNUSED,
                         bool reduce_all,
                         DenseTensor* in_grad) {
  if (in_grad && in_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(in_grad);
    return;
  }
  std::vector<int64_t> axis;
  axis.reserve(axis_in.size());
  std::for_each(axis_in.begin(), axis_in.end(), [&axis](const int& t) {
    axis.push_back(static_cast<int64_t>(t));
  });
  dev_ctx.template Alloc<T>(in_grad);

  reduce_all = recompute_reduce_all(in, axis, reduce_all);

  if (FLAGS_use_accuracy_compatible_kernel) {
    LogsumexpGradImpl<T, Context, LogsumexpGradCompatibleFunctor<T>>(
        dev_ctx, in, out, out_grad, axis, reduce_all, in_grad);
  } else {
    LogsumexpGradImpl<T, Context, LogsumexpGradFunctor<T>>(
        dev_ctx, in, out, out_grad, axis, reduce_all, in_grad);
  }
}

}  // namespace phi
