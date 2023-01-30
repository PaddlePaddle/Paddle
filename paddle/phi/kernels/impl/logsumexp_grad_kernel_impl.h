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

<<<<<<< HEAD
#include "paddle/phi/common/amp_type_traits.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"
#include "paddle/phi/kernels/logsumexp_grad_kernel.h"

namespace phi {

<<<<<<< HEAD
template <typename T>
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    auto x_mt = (*x).template cast<MT>();
    auto y_mt = (*y).template cast<MT>();
    auto dy_mt = (*dy).template cast<MT>();
    dx->device(place) =
        (dy_mt.broadcast(dim) * (x_mt - y_mt.broadcast(dim)).exp())
            .template cast<T>();
=======
    dx->device(place) = dy->broadcast(dim) * (*x - y->broadcast(dim)).exp();
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
  reduce_all = recompute_reduce_all(in, axis, reduce_all);
=======
  const auto input_dim_size = in.dims().size();
  reduce_all |= (static_cast<const int>(axis.size()) == input_dim_size);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  if (reduce_all) {
    auto x = phi::EigenVector<T>::Flatten(in);
    auto y = phi::EigenVector<T>::Flatten(out);
    auto dy = phi::EigenVector<T>::Flatten(out_grad);
    auto dx = phi::EigenVector<T>::Flatten(*in_grad);
    auto& place = *dev_ctx.eigen_device();
    auto broadcast_dim = Eigen::array<int, 1>({{static_cast<int>(in.numel())}});
<<<<<<< HEAD
    LogsumexpGradFunctor<T>()(
        place, &x, &y, &dx, &dy, broadcast_dim, broadcast_dim[0]);
  } else {
    int rank = in.dims().size();
    LogsumexpGradFunctor<T> functor;
=======
    LogsumexpGradFunctor()(
        place, &x, &y, &dx, &dy, broadcast_dim, broadcast_dim[0]);
  } else {
    int rank = in.dims().size();
    LogsumexpGradFunctor functor;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    std::vector<int32_t> axis32;
    axis32.reserve(axis.size());
    std::for_each(axis.begin(), axis.end(), [&axis32](const int64_t& t) {
      axis32.push_back(t);
    });
    switch (rank) {
      case 1:
<<<<<<< HEAD
        phi::funcs::ReduceGradFunctor<Context, T, 1, LogsumexpGradFunctor<T>>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 2:
        phi::funcs::ReduceGradFunctor<Context, T, 2, LogsumexpGradFunctor<T>>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 3:
        phi::funcs::ReduceGradFunctor<Context, T, 3, LogsumexpGradFunctor<T>>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      case 4:
        phi::funcs::ReduceGradFunctor<Context, T, 4, LogsumexpGradFunctor<T>>(
            dev_ctx, in, out, out_grad, in_grad, functor, axis32);
        break;
      default:
        PADDLE_THROW(phi::errors::Unimplemented(
            "Unsupported dimensions, please keep maximum dimensions of input "
            "data less than 4."));
        break;
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    }
  }
}

}  // namespace phi
