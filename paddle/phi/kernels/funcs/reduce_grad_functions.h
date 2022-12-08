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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
namespace phi {

namespace funcs {

// This ReduceGradFunctor is only the CPU implement.
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

  int broad_cast_times = 1;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) {
      dims_ref[i] = x_rank + dims_ref[i];
    }
    reduced_dims_v[dims_ref[i]] = 1;
    broadcast_dim[dims_ref[i]] = x_dims[dims_ref[i]];
    broad_cast_times *= x_dims[dims_ref[i]];
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
          broad_cast_times);
}

inline void GetOriginDimFromShuffled(const DDim& src_dim,
                                     const std::vector<int>& dims,
                                     std::vector<int>* origin_dim) {
  DDim shuffled_dims(src_dim);
  size_t n = src_dim.size();
  std::vector<int> perm_axis(n);
  std::vector<int64_t> dims_64{dims.begin(), dims.end()};
  GetShuffledDim(src_dim, &shuffled_dims, dims_64, &perm_axis);
  for (size_t i = 0; i < n; ++i) {
    (*origin_dim)[perm_axis[i]] = i;
  }
}

template <typename Context, typename T, typename Functor>
void HandleLargeDimGrad(const Context& dev_ctx,
                        const DenseTensor* x,
                        const DenseTensor* out,
                        const DenseTensor* dout,
                        DenseTensor* dx,
                        Functor functor,
                        const std::vector<int>& dims) {
  const int64_t unreduced = out->numel();
  const int64_t reduced = x->numel() / unreduced;
  DDim out_dim(out->dims());
  DDim x_dim(x->dims());
  // transpose and reshape X
  DenseTensor shuffled_x;
  std::vector<int64_t> dims_64{dims.begin(), dims.end()};
  GetShuffledInput<Context, T>(dev_ctx, *x, &shuffled_x, dims_64);
  DDim shuffled_dim = shuffled_x.dims();
  shuffled_x.Resize({unreduced, reduced});
  // reshape dX {unreduced, reduced}
  dx->Resize({unreduced, reduced});
  ReduceGradFunctor<Context, T, 2, Functor>(
      dev_ctx, shuffled_x, *out, *dout, dx, functor, {1});
  // transpose dX
  std::vector<int> origin_axis(x_dim.size());
  GetOriginDimFromShuffled(x_dim, dims, &origin_axis);
  DenseTensor dx_tmp;
  paddle::framework::TensorCopy(*dx, dev_ctx.GetPlace(), &dx_tmp);
  dx_tmp.Resize(shuffled_dim);
  dx->Resize(x_dim);
  phi::funcs::TransposeNormal<Context, T> trans;
  trans(dev_ctx, dx_tmp, dx, origin_axis);
}

// Only for CPU
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
    // *dev_ctx.eigen_device();
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
            dev_ctx, *input0, *input1, *input2, output, functor, dims);
        break;
      case 2:
        ReduceGradFunctor<Context, T, 2, Functor>(
            dev_ctx, *input0, *input1, *input2, output, functor, dims);
        break;
      case 3:
        ReduceGradFunctor<Context, T, 3, Functor>(
            dev_ctx, *input0, *input1, *input2, output, functor, dims);
        break;
      case 4:
        ReduceGradFunctor<Context, T, 4, Functor>(
            dev_ctx, *input0, *input1, *input2, output, functor, dims);
        break;
      case 5:
        ReduceGradFunctor<Context, T, 5, Functor>(
            dev_ctx, *input0, *input1, *input2, output, functor, dims);
        break;
      case 6:
        ReduceGradFunctor<Context, T, 6, Functor>(
            dev_ctx, *input0, *input1, *input2, output, functor, dims);
        break;
      default:
        HandleLargeDimGrad<Context, T, Functor>(
            dev_ctx, input0, input1, input2, output, functor, dims);
        break;
    }
  }
}

}  // namespace funcs

}  // namespace phi
