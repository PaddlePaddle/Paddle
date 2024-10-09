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
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"
#include "paddle/phi/kernels/logsumexp_grad_kernel.h"

namespace phi {

template <typename T, class Enable = void>
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
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    auto x_mt = (*x).template cast<MT>();
    auto y_mt = (*y).template cast<MT>();
    auto dy_mt = (*dy).template cast<MT>();
    dx->device(place) =
        (dy_mt.broadcast(dim) * (x_mt - y_mt.broadcast(dim)).exp())
            .template cast<T>();
  }
};

template <typename T>
struct LogsumexpGradFunctor<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type> {
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
    auto x_u = (*x).template cast<float>();
    auto y_u = (*y).template cast<float>();
    auto dy_u = (*dy).template cast<float>();
    dx->device(place) = (dy_u.broadcast(dim) * (x_u - y_u.broadcast(dim)).exp())
                            .template cast<float>();
  }
};

template <typename T, typename Context>
void LogsumexpGradKernel(const Context& dev_ctx,
                         const DenseTensor& in,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis_in,
                         bool keepdim UNUSED,
                         bool reduce_all,
                         DenseTensor* in_grad) {
  std::vector<int64_t> axis;
  axis.reserve(axis_in.size());
  std::for_each(axis_in.begin(), axis_in.end(), [&axis](const int& t) {
    axis.push_back(static_cast<int64_t>(t));
  });
  using U = typename std::conditional_t<std::is_integral<T>::value, float, T>;
  dev_ctx.template Alloc<U>(in_grad);

  reduce_all = recompute_reduce_all(in, axis, reduce_all);

  if (reduce_all) {
    auto x = phi::EigenVector<T>::Flatten(in);
    auto y = phi::EigenVector<U>::Flatten(out);
    auto dy = phi::EigenVector<U>::Flatten(out_grad);
    auto dx = phi::EigenVector<U>::Flatten(*in_grad);
    auto& place = *dev_ctx.eigen_device();
    auto broadcast_dim = Eigen::array<int, 1>({{static_cast<int>(in.numel())}});
    LogsumexpGradFunctor<T>()(
        place, &x, &y, &dx, &dy, broadcast_dim, broadcast_dim[0]);
  } else {
    int rank = in.dims().size();
    LogsumexpGradFunctor<U> functor;
    std::vector<int32_t> axis32;
    axis32.reserve(axis.size());
    std::for_each(axis.begin(), axis.end(), [&axis32](const int64_t& t) {
      axis32.push_back(t);
    });
    if (std::is_integral<T>::value) {
      auto tmp_in = phi::Cast<T, Context>(dev_ctx, in, phi::DataType::FLOAT32);
      switch (rank) {
        case 1:
          phi::funcs::ReduceGradFunctor<Context, U, 1, LogsumexpGradFunctor<U>>(
              dev_ctx, tmp_in, out, out_grad, in_grad, functor, axis32);
          break;
        case 2:
          phi::funcs::ReduceGradFunctor<Context, U, 2, LogsumexpGradFunctor<U>>(
              dev_ctx, tmp_in, out, out_grad, in_grad, functor, axis32);
          break;
        case 3:
          phi::funcs::ReduceGradFunctor<Context, U, 3, LogsumexpGradFunctor<U>>(
              dev_ctx, tmp_in, out, out_grad, in_grad, functor, axis32);
          break;
        case 4:
          phi::funcs::ReduceGradFunctor<Context, U, 4, LogsumexpGradFunctor<U>>(
              dev_ctx, tmp_in, out, out_grad, in_grad, functor, axis32);
          break;
        default:
          PADDLE_THROW(common::errors::Unimplemented(
              "Unsupported dimensions, please keep maximum dimensions of input "
              "data less than 4."));
          break;
      }
    } else {
      switch (rank) {
        case 1:
          phi::funcs::ReduceGradFunctor<Context, U, 1, LogsumexpGradFunctor<U>>(
              dev_ctx, in, out, out_grad, in_grad, functor, axis32);
          break;
        case 2:
          phi::funcs::ReduceGradFunctor<Context, U, 2, LogsumexpGradFunctor<U>>(
              dev_ctx, in, out, out_grad, in_grad, functor, axis32);
          break;
        case 3:
          phi::funcs::ReduceGradFunctor<Context, U, 3, LogsumexpGradFunctor<U>>(
              dev_ctx, in, out, out_grad, in_grad, functor, axis32);
          break;
        case 4:
          phi::funcs::ReduceGradFunctor<Context, U, 4, LogsumexpGradFunctor<U>>(
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
}

}  // namespace phi
