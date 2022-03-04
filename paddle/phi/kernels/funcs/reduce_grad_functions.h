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

#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
// #include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
namespace phi {

template <typename Context, typename T, size_t D, typename Functor>
void ReduceGradFunctor(const Context& dev_ctx,
                       const DenseTensor& input0,
                       const DenseTensor& input1,
                       const DenseTensor& input2,
                       DenseTensor* output,
                       Functor functor,
                       const std::vector<int>& dims) {
  auto x = phi::EigenTensor<T, D>::From(input0);
  auto x_grad = phi::EigenTensor<T, D>::From(*output);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto x_dims = input0.dims();
  auto reduced_dims_v = phi::vectorize(x_dims);
  std::vector<int> dims_ref = dims;
  Eigen::array<int, D> broadcast_dim;
  for (size_t i = 0; i < D; ++i) broadcast_dim[i] = 1;

  int broad_cats_times = 1;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) {
      dims_ref[i] = x_rank + dims_ref[i];
    }
    reduced_dims_v[dims_ref[i]] = 1;
    broadcast_dim[dims_ref[i]] = x_dims[dims_ref[i]];
    broad_cats_times *= x_dims[dims_ref[i]];
  }
  auto reduced_dims = phi::make_ddim(reduced_dims_v);
  auto x_reduce = EigenTensor<T, D>::From(input1, reduced_dims);
  auto x_reduce_grad = EigenTensor<T, D>::From(input2, reduced_dims);

  auto& place = *dev_ctx.eigen_device();

  functor(place,
          &x,
          &x_reduce,
          &x_grad,
          &x_reduce_grad,
          broadcast_dim,
          broad_cats_times);
}

template <typename Context, typename T, typename Functor>
void LaunchReduceGradKernel(const Context& dev_ctx,
                            const DenseTensor* input0,
                            const DenseTensor* input1,
                            const DenseTensor* input2,
                            DenseTensor* output,
                            Functor functor,
                            const std::vector<int>& dims,
                            bool reduce_all = false) {
  if (reduce_all) {
    auto x = phi::EigenVector<T>::Flatten(*input0);
    auto x_reduce = phi::EigenVector<T>::Flatten(*input1);
    auto x_reduce_grad = phi::EigenVector<T>::Flatten(*input2);
    auto x_grad = phi::EigenVector<T>::Flatten(*output);
    auto& place = *dev_ctx.eigen_device();
    // *dev_ctx.template device_context<Context>().eigen_device();
    auto broadcast_dim =
        Eigen::array<int, 1>({{static_cast<int>(input0->numel())}});
    functor(place,
            &x,
            &x_reduce,
            &x_grad,
            &x_reduce_grad,
            broadcast_dim,
            broadcast_dim[0]);
  } else {
    int rank = input0->dims().size();
    switch (rank) {
      case 1:
        ReduceGradFunctor<Context, T, 1, Functor>(
            dev_ctx.template device_context<Context>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 2:
        ReduceGradFunctor<Context, T, 2, Functor>(
            dev_ctx.template device_context<Context>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 3:
        ReduceGradFunctor<Context, T, 3, Functor>(
            dev_ctx.template device_context<Context>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 4:
        ReduceGradFunctor<Context, T, 4, Functor>(
            dev_ctx.template device_context<Context>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 5:
        ReduceGradFunctor<Context, T, 5, Functor>(
            dev_ctx.template device_context<Context>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      case 6:
        ReduceGradFunctor<Context, T, 6, Functor>(
            dev_ctx.template device_context<Context>(),
            *input0,
            *input1,
            *input2,
            output,
            functor,
            dims);
        break;
      default:
        HandleLargeDimGrad<Context, T, Functor>(
            dev_ctx, input0, input1, input2, output, functor, dims);
        break;
    }
  }
}

}  // namespace phi
