/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using ETensor = phi::EigenTensor<T, D, MajorType, IndexType>;

template <int Rank>
static void GetBraodcastDims(const phi::DDim& x_dims,
                             const phi::DDim& y_dims,
                             Eigen::DSizes<int, Rank>* x_bcast_dims,
                             Eigen::DSizes<int, Rank>* y_bcast_dims) {
  int bcast_dims_remainder = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    if (x_dims[i] >= y_dims[i]) {
      (*x_bcast_dims)[i] = 1;
      (*y_bcast_dims)[i] = x_dims[i] / y_dims[i];
      bcast_dims_remainder += x_dims[i] % y_dims[i];
    } else {
      (*y_bcast_dims)[i] = 1;
      (*x_bcast_dims)[i] = y_dims[i] / x_dims[i];
      bcast_dims_remainder += y_dims[i] % x_dims[i];
    }
  }
  PADDLE_ENFORCE_EQ(bcast_dims_remainder,
                    0,
                    phi::errors::PreconditionNotMet(
                        "The input tensor of Op(dist) could not be broadcast, "
                        "X's shape is [%s], Y's shape is [%s].",
                        x_dims,
                        y_dims));
}

static phi::DDim GetNewDims(const phi::DDim& in_dims, int rank) {
  std::vector<int64_t> new_dims_vec(rank);
  if (in_dims.size() < rank) {
    for (int i = 0; i < rank - in_dims.size(); ++i) {
      new_dims_vec[i] = 1;
    }
    for (int i = 0; i < in_dims.size(); ++i) {
      new_dims_vec[i + rank - in_dims.size()] = in_dims[i];
    }
  } else {
    new_dims_vec = vectorize(in_dims);
  }
  return phi::make_ddim(new_dims_vec);
}

template <typename Context, typename T, int Rank>
static void DistGradFunction(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             const DenseTensor& out,
                             const DenseTensor& out_grad,
                             float p,
                             DenseTensor* x_grad,
                             DenseTensor* y_grad) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto out_dims = out.dims();

  phi::DDim x_new_dims = GetNewDims(x_dims, Rank);
  phi::DDim y_new_dims = GetNewDims(y_dims, Rank);
  phi::DDim out_new_dims = GetNewDims(out_dims, Rank);
  auto x_t = ETensor<T, Rank>::From(x, x_new_dims);
  auto y_t = ETensor<T, Rank>::From(y, y_new_dims);
  auto out_t = ETensor<T, Rank>::From(out, out_new_dims);

  Eigen::DSizes<int, Rank> x_bcast_dims;
  Eigen::DSizes<int, Rank> y_bcast_dims;
  Eigen::DSizes<int, Rank> out_bcast_dims;

  GetBraodcastDims<Rank>(x_new_dims, y_new_dims, &x_bcast_dims, &y_bcast_dims);
  std::vector<int64_t> new_dims_vec(Rank);
  for (int i = 0; i < Rank; ++i) {
    new_dims_vec[i] = std::max(x_new_dims[i], y_new_dims[i]);
    out_bcast_dims[i] = new_dims_vec[i];
  }
  phi::DDim new_dims = phi::make_ddim(new_dims_vec);

  auto& place = *dev_ctx.eigen_device();
  auto out_grad_t = ETensor<T, Rank>::From(out_grad, out_new_dims);
  DenseTensor grad;
  grad.Resize(new_dims);
  dev_ctx.template Alloc<T>(&grad);
  auto grad_t = ETensor<T, Rank>::From(grad);

  auto x_minux_y = x_t.broadcast(x_bcast_dims) - y_t.broadcast(y_bcast_dims);
  auto x_minux_y_abs = x_minux_y.abs();
  auto sign =
      (x_minux_y > static_cast<T>(0)).template cast<T>() * static_cast<T>(1.0) +
      (x_minux_y < static_cast<T>(0)).template cast<T>() * static_cast<T>(-1.0);
  T epsilon = static_cast<T>(1.0e-10f);

  // 1: Lp-norm(z), z = x-y, compute dz
  if (p == 0) {
    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, &grad, static_cast<T>(0));
  } else if (p == INFINITY || p == -INFINITY) {
    // p=inf or -inf, Lp-norm = |z_i|, the j-th element of dz tends to 0 if
    // j!=i, or equals to sign(z_i) * dout if j=i.
    if (paddle::platform::is_cpu_place(dev_ctx.GetPlace())) {
      grad_t.device(place) = (x_minux_y_abs == out_t.broadcast(out_bcast_dims))
                                 .template cast<T>() *
                             sign.eval() * out_grad_t.broadcast(out_bcast_dims);
    } else {
      grad_t.device(place) = (x_minux_y_abs == out_t.broadcast(out_bcast_dims))
                                 .template cast<T>() *
                             sign * out_grad_t.broadcast(out_bcast_dims);
    }
  } else {
    // dz = pow(abs(x-y)/out, p-1) * sign(x-y) * dout
    if (paddle::platform::is_cpu_place(dev_ctx.GetPlace())) {
      grad_t.device(place) =
          (x_minux_y_abs / (out_t + epsilon).broadcast(out_bcast_dims))
              .pow(p - 1) *
          sign.eval() * out_grad_t.broadcast(out_bcast_dims);
    } else {
      grad_t.device(place) =
          (x_minux_y_abs / (out_t + epsilon).broadcast(out_bcast_dims))
              .pow(p - 1) *
          sign * out_grad_t.broadcast(out_bcast_dims);
    }
  }

  Eigen::DSizes<int, Rank * 2> x_reshape_dims;
  Eigen::DSizes<int, Rank * 2> y_reshape_dims;
  Eigen::DSizes<int, Rank> reduce_dims;
  for (int i = 0; i < x_new_dims.size(); ++i) {
    x_reshape_dims[2 * i] = x_bcast_dims[i];
    x_reshape_dims[2 * i + 1] = x_new_dims[i];
    y_reshape_dims[2 * i] = y_bcast_dims[i];
    y_reshape_dims[2 * i + 1] = y_new_dims[i];
    reduce_dims[i] = 2 * i;
  }

  // 2: if x or y is broadcasted in forward function,
  // the grad need to be sum along the broadcasted dimensions
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    auto x_grad_t = ETensor<T, Rank>::From(*x_grad, x_new_dims);
    x_grad_t.device(place) = grad_t.reshape(x_reshape_dims)
                                 .sum(reduce_dims)
                                 .reshape(x_grad_t.dimensions());
  }
  if (y_grad) {
    dev_ctx.template Alloc<T>(y_grad);
    auto y_grad_t = ETensor<T, Rank>::From(*y_grad, y_new_dims);
    y_grad_t.device(place) = -grad_t.reshape(y_reshape_dims)
                                  .sum(reduce_dims)
                                  .reshape(y_grad_t.dimensions());
  }
}

template <typename T, typename Context>
void DistGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const DenseTensor& out,
                    const DenseTensor& out_grad,
                    float p,
                    DenseTensor* x_grad,
                    DenseTensor* y_grad) {
  auto x_rank = x.dims().size();
  auto y_rank = y.dims().size();
  auto rank = std::max(x_rank, y_rank);
  PADDLE_ENFORCE_LE(rank,
                    6,
                    phi::errors::Unimplemented(
                        "Op(dist) only support tensors with no more than 6 "
                        "dimensions, but X's rank is %d, Y's rank is %d.",
                        x_rank,
                        y_rank));
  switch (rank) {
    case 1:
      DistGradFunction<Context, T, 1>(
          dev_ctx, x, y, out, out_grad, p, x_grad, y_grad);
      break;
    case 2:
      DistGradFunction<Context, T, 2>(
          dev_ctx, x, y, out, out_grad, p, x_grad, y_grad);
      break;
    case 3:
      DistGradFunction<Context, T, 3>(
          dev_ctx, x, y, out, out_grad, p, x_grad, y_grad);
      break;
    case 4:
      DistGradFunction<Context, T, 4>(
          dev_ctx, x, y, out, out_grad, p, x_grad, y_grad);
      break;
    case 5:
      DistGradFunction<Context, T, 5>(
          dev_ctx, x, y, out, out_grad, p, x_grad, y_grad);
      break;
    case 6:
      DistGradFunction<Context, T, 6>(
          dev_ctx, x, y, out, out_grad, p, x_grad, y_grad);
      break;
  }
}

}  // namespace phi
