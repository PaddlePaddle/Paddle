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

#include "paddle/pten/kernels/lerp_kernel.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"

namespace pten {

static framework::DDim ExtendDims2Rank(const framework::DDim& in_dims,
                                       int rank) {
  if (in_dims.size() == rank) {
    return in_dims;
  }
  std::vector<int64_t> shapes(rank, 1);
  for (int i = in_dims.size() - 1, j = rank - 1; i >= 0; --i, --j) {
    shapes[j] = in_dims[i];
  }
  return framework::make_ddim(shapes);
}

template <size_t D>
static void GetBroadcastDims(const framework::DDim& in_dims,
                             const framework::DDim& out_dims,
                             Eigen::DSizes<int, D>* bcast_dims) {
  for (size_t i = 0; i < D; ++i) {
    if (in_dims[i] == out_dims[i]) {
      (*bcast_dims)[i] = 1;
    } else {
      (*bcast_dims)[i] = std::max(in_dims[i], out_dims[i]);
    }
  }
}

template <typename Context, typename T, size_t D>
static void LerpFunction(const Context& ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& weight,
                         DenseTensor* out) {
  // auto x = ctx.Input<framework::Tensor>("X");
  // auto y = ctx.Input<framework::Tensor>("Y");
  // auto w = ctx.Input<framework::Tensor>("Weight");
  // auto out = ctx.Output<framework::Tensor>("Out");

  out->mutable_data<T>(ctx.GetPlace());
  // ctx.Alloc<T>(out);

  auto out_dims = out->dims();
  auto x_dims = ExtendDims2Rank(x.dims(), D);
  auto y_dims = ExtendDims2Rank(y.dims(), D);
  auto w_dims = ExtendDims2Rank(weight.dims(), D);
  Eigen::DSizes<int, D> x_bcast_dims;
  Eigen::DSizes<int, D> y_bcast_dims;
  Eigen::DSizes<int, D> w_bcast_dims;
  GetBroadcastDims<D>(x_dims, out_dims, &x_bcast_dims);
  GetBroadcastDims<D>(y_dims, out_dims, &y_bcast_dims);
  GetBroadcastDims<D>(w_dims, out_dims, &w_bcast_dims);

  auto eigen_x = pten::EigenTensor<T, D>::From(x, x_dims);
  auto eigen_y = pten::EigenTensor<T, D>::From(y, y_dims);
  auto eigen_w = pten::EigenTensor<T, D>::From(weight, w_dims);
  auto eigen_out = pten::EigenTensor<T, D>::From(*out);

  // auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  auto& place = *ctx.eigen_device();
  eigen_out.device(place) =
      eigen_x.broadcast(x_bcast_dims) +
      eigen_w.broadcast(w_bcast_dims) *
          (eigen_y.broadcast(y_bcast_dims) - eigen_x.broadcast(x_bcast_dims));
}

template <typename T, typename Context>
void LerpKernel(const Context& ctx,
                const DenseTensor& x,
                const DenseTensor& y,
                const DenseTensor& weight,
                DenseTensor* out) {
  int rank = out->dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1,
      pten::errors::InvalidArgument(
          "The number of dimensions for LerpOp must be "
          "greater than or equal to 1, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      pten::errors::InvalidArgument(
          "The number of dimensions for LerpOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));
  switch (rank) {
    case 1:
      LerpFunction<Context, T, 1>(ctx, x, y, weight, out);
      break;
    case 2:
      LerpFunction<Context, T, 2>(ctx, x, y, weight, out);
      break;
    case 3:
      LerpFunction<Context, T, 3>(ctx, x, y, weight, out);
      break;
    case 4:
      LerpFunction<Context, T, 4>(ctx, x, y, weight, out);
      break;
    case 5:
      LerpFunction<Context, T, 5>(ctx, x, y, weight, out);
      break;
    case 6:
      LerpFunction<Context, T, 6>(ctx, x, y, weight, out);
      break;
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(lerp, CPU, ALL_LAYOUT, pten::LerpKernel, float, double) {}
