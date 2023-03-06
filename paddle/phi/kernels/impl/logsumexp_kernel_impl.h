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
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/logsumexp_kernel.h"

namespace phi {

#define HANDLE_DIM(NDIM, RDIM)                                         \
  if (ndim == NDIM && rdim == RDIM) {                                  \
    funcs::ReduceFunctor<Context, T, NDIM, RDIM, LogsumexpFunctor<T>>( \
        dev_ctx, x, out, axis, keepdim);                               \
  }

template <typename T>
struct LogsumexpFunctor {
  template <typename Context, typename X, typename Y, typename Dim>
  void operator()(const Context& place, X* x, Y* y, const Dim& dim) {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    auto x_dim = x->dimensions();
    auto t_dim = x_dim;
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      t_dim[dim[i]] = 1;
    }

    auto r_dim = x_dim;
    for (int i = 0; i < static_cast<int>(r_dim.size()); i++) {
      r_dim[i] = 1;
    }
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      r_dim[dim[i]] = x_dim[dim[i]];
    }

    auto x_mt = (*x).template cast<MT>();
    auto y_dim = y->dimensions();
    auto x_max = x_mt.maximum(dim).eval();
    y->device(place) =
        (x_max +
         (x_mt - x_max.reshape(t_dim).broadcast(r_dim)).exp().sum(dim).log())
            .reshape(y_dim)
            .template cast<T>();
  }
};

template <typename T, typename Context>
void LogsumexpKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keepdim,
                     bool reduce_all,
                     DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  reduce_all = recompute_reduce_all(x, axis, reduce_all);

  auto x_dim = x.dims();
  for (int i = 0; i < x_dim.size(); i++) {
    PADDLE_ENFORCE_LT(0,
                      x_dim[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
  }
  if (reduce_all) {
    // Flatten and reduce 1-D tensor
    auto input = phi::EigenVector<T>::Flatten(x);
    auto output = phi::EigenScalar<T>::From(*out);
    auto& place = *dev_ctx.eigen_device();
    auto reduce_dim = Eigen::array<int, 1>({{0}});
    LogsumexpFunctor<T>()(place, &input, &output, reduce_dim);
  } else {
    int ndim = x.dims().size();
    int rdim = axis.size();
    if (ndim > 4) {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported dimensions, please keep maximum dimensions of input "
          "data less than 4."));
    }

    // comments for accelerating compiling temporarily.
    // HANDLE_DIM(6, 5);
    // HANDLE_DIM(6, 4);
    // HANDLE_DIM(6, 3);
    // HANDLE_DIM(6, 2);
    // HANDLE_DIM(6, 1);
    // HANDLE_DIM(5, 4);
    // HANDLE_DIM(5, 3);
    // HANDLE_DIM(5, 2);
    // HANDLE_DIM(5, 1);
    HANDLE_DIM(4, 3);
    HANDLE_DIM(4, 2);
    HANDLE_DIM(4, 1);
    HANDLE_DIM(3, 2);
    HANDLE_DIM(3, 1);
    HANDLE_DIM(2, 1);
  }
}

}  // namespace phi
