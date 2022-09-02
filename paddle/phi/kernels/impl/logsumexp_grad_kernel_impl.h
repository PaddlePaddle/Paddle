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

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"
#include "paddle/phi/kernels/logsumexp_grad_kernel.h"

namespace phi {

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
                  int size) {
    dx->device(place) = dy->broadcast(dim) * (*x - y->broadcast(dim)).exp();
  }
};

template <typename T, typename Context>
void LogsumexpGradKernel(const Context& dev_ctx,
                         const DenseTensor& in,
                         const DenseTensor& out,
                         const DenseTensor& out_grad,
                         const std::vector<int64_t>& axis,
                         bool keepdim,
                         bool reduce_all,
                         DenseTensor* in_grad) {
  dev_ctx.template Alloc<T>(in_grad);

  const auto input_dim_size = in.dims().size();
  reduce_all |= (static_cast<const int>(axis.size()) == input_dim_size);

  if (reduce_all) {
    auto x = phi::EigenVector<T>::Flatten(in);
    auto y = phi::EigenVector<T>::Flatten(out);
    auto dy = phi::EigenVector<T>::Flatten(out_grad);
    auto dx = phi::EigenVector<T>::Flatten(*in_grad);
    auto& place = *dev_ctx.eigen_device();
    auto broadcast_dim = Eigen::array<int, 1>({{static_cast<int>(in.numel())}});
    LogsumexpGradFunctor()(
        place, &x, &y, &dx, &dy, broadcast_dim, broadcast_dim[0]);
  } else {
    int rank = in.dims().size();
    LogsumexpGradFunctor functor;
    std::vector<int32_t> axis32;
    axis32.reserve(axis.size());
    std::for_each(axis.begin(), axis.end(), [&axis32](const int64_t& t) {
      axis32.push_back(t);
    });
    switch (rank) {
      case 1:
        phi::funcs::ReduceGradFunctor<Context, T, 1, LogsumexpGradFunctor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 2:
        phi::funcs::ReduceGradFunctor<Context, T, 2, LogsumexpGradFunctor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 3:
        phi::funcs::ReduceGradFunctor<Context, T, 3, LogsumexpGradFunctor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 4:
        phi::funcs::ReduceGradFunctor<Context, T, 4, LogsumexpGradFunctor>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
    }
  }
}

}  // namespace phi
