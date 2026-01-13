// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/ddim.h"
#include "paddle/common/errors.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename Context, typename T, size_t D>
static void AddcmulGradFunction(const Context& dev_ctx,
                                const DenseTensor& input UNUSED,
                                const DenseTensor& tensor1,
                                const DenseTensor& tensor2,
                                const DenseTensor& out_grad,
                                const Scalar& value,
                                DenseTensor* input_grad,
                                DenseTensor* t1_grad,
                                DenseTensor* t2_grad) {
  auto& dout = out_grad;
  auto* dx = input_grad;
  auto* dt1 = t1_grad;
  auto* dt2 = t2_grad;

  // Output dims (broadcasted dims) is dout dims
  // But wait, dout dims might be same as output dims.
  // We need the broadcasted shape `out_dims` to properly reduce.
  // Usually dout has same shape as forward output.
  auto out_dims = out_grad.dims();

  DDim dx_dims, dt1_dims, dt2_dims;

  auto t1_dims = funcs::ExtendDims2Rank(tensor1.dims(), D);
  auto t2_dims = funcs::ExtendDims2Rank(tensor2.dims(), D);
  auto g_dims = funcs::ExtendDims2Rank(out_grad.dims(), D);

  Eigen::DSizes<int, D> dx_bcast_dims;
  Eigen::DSizes<int, D> dt1_bcast_dims;
  Eigen::DSizes<int, D> dt2_bcast_dims;
  Eigen::DSizes<int, D> t1_bcast_dims;
  Eigen::DSizes<int, D> t2_bcast_dims;

  // Calculate broadcast dims for inputs vs output
  if (dx) {
    dx_dims = funcs::ExtendDims2Rank(dx->dims(), D);
    funcs::GetBroadcastDims<D>(dx_dims, out_dims, &dx_bcast_dims);
  }
  if (dt1) {
    dt1_dims = funcs::ExtendDims2Rank(dt1->dims(), D);
    funcs::GetBroadcastDims<D>(dt1_dims, out_dims, &dt1_bcast_dims);
  }
  if (dt2) {
    dt2_dims = funcs::ExtendDims2Rank(dt2->dims(), D);
    funcs::GetBroadcastDims<D>(dt2_dims, out_dims, &dt2_bcast_dims);
  }

  // Need broadcast dims for t1 and t2 to use in expression
  funcs::GetBroadcastDims<D>(t1_dims, out_dims, &t1_bcast_dims);
  funcs::GetBroadcastDims<D>(t2_dims, out_dims, &t2_bcast_dims);

  auto eigen_t1 = phi::EigenTensor<T, D>::From(tensor1, t1_dims);
  auto eigen_t2 = phi::EigenTensor<T, D>::From(tensor2, t2_dims);
  auto eigen_dout = phi::EigenTensor<T, D>::From(dout, g_dims);

  Eigen::DSizes<int, D * 2> dx_reshape_dims;
  Eigen::DSizes<int, D * 2> dt1_reshape_dims;
  Eigen::DSizes<int, D * 2> dt2_reshape_dims;
  Eigen::DSizes<int, D> reduce_dims;

  for (int i = 0; i < D; ++i) {
    if (dx) {
      dx_reshape_dims[2 * i] = dx_bcast_dims[i];
      dx_reshape_dims[2 * i + 1] = dx_dims[i];
    }
    if (dt1) {
      dt1_reshape_dims[2 * i] = dt1_bcast_dims[i];
      dt1_reshape_dims[2 * i + 1] = dt1_dims[i];
    }
    if (dt2) {
      dt2_reshape_dims[2 * i] = dt2_bcast_dims[i];
      dt2_reshape_dims[2 * i + 1] = dt2_dims[i];
    }
    reduce_dims[i] = 2 * i;
  }

  auto& place = *dev_ctx.eigen_device();

  // d(input) = sum(dout)
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    auto eigen_dx = phi::EigenTensor<T, D>::From(*dx, dx_dims);
    eigen_dx.device(place) =
        eigen_dout
            .broadcast(
                Eigen::DSizes<int, D>())  // dout is already broadcasted shape?
                                          // No, dout is output shape. Wait,
                                          // eigen_dout is From(dout, g_dims).
                                          // g_dims is extended dout dims. We
                                          // treat dout as fully broadcasted
                                          // result. We reshape it to split
                                          // broadcasted dimensions and reduce.
            .reshape(dx_reshape_dims)
            .sum(reduce_dims)
            .reshape(eigen_dx.dimensions());
  }

  using MPType = typename dtype::MPTypeTrait<T>::Type;

  // d(t1) = sum(dout * value * t2)
  if (dt1) {
    dev_ctx.template Alloc<T>(dt1);
    auto eigen_dt1 = phi::EigenTensor<T, D>::From(*dt1, dt1_dims);
    auto expr = eigen_dout.template cast<MPType>() *
                eigen_t2.broadcast(t2_bcast_dims).template cast<MPType>() *
                static_cast<MPType>(value.to<T>());

    eigen_dt1.device(place) = expr.reshape(dt1_reshape_dims)
                                  .sum(reduce_dims)
                                  .reshape(eigen_dt1.dimensions())
                                  .template cast<T>();
  }

  // d(t2) = sum(dout * value * t1)
  if (dt2) {
    dev_ctx.template Alloc<T>(dt2);
    auto eigen_dt2 = phi::EigenTensor<T, D>::From(*dt2, dt2_dims);
    auto expr = eigen_dout.template cast<MPType>() *
                eigen_t1.broadcast(t1_bcast_dims).template cast<MPType>() *
                static_cast<MPType>(value.to<T>());

    eigen_dt2.device(place) = expr.reshape(dt2_reshape_dims)
                                  .sum(reduce_dims)
                                  .reshape(eigen_dt2.dimensions())
                                  .template cast<T>();
  }
}

template <typename Context, typename T>
static void AddcmulGradFunctionZero(const Context& dev_ctx,
                                    const DenseTensor& input UNUSED,
                                    const DenseTensor& tensor1,
                                    const DenseTensor& tensor2,
                                    const DenseTensor& out_grad,
                                    const Scalar& value,
                                    DenseTensor* input_grad,
                                    DenseTensor* t1_grad,
                                    DenseTensor* t2_grad) {
  // Zero dim logic (scalar)
  auto dim = ::common::make_ddim(std::vector<int64_t>(1, 1));
  auto eigen_t1 = phi::EigenTensor<T, 1>::From(tensor1, dim);
  auto eigen_t2 = phi::EigenTensor<T, 1>::From(tensor2, dim);
  auto eigen_dout = phi::EigenTensor<T, 1>::From(out_grad, dim);

  using MPType = typename dtype::MPTypeTrait<T>::Type;
  auto& place = *dev_ctx.eigen_device();

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    auto eigen_dx = phi::EigenTensor<T, 1>::From(*input_grad, dim);
    eigen_dx.device(place) = eigen_dout;
  }

  if (t1_grad) {
    dev_ctx.template Alloc<T>(t1_grad);
    auto eigen_dt1 = phi::EigenTensor<T, 1>::From(*t1_grad, dim);
    eigen_dt1.device(place) =
        (eigen_dout.template cast<MPType>() * eigen_t2.template cast<MPType>() *
         static_cast<MPType>(value.to<T>()))
            .template cast<T>();
  }

  if (t2_grad) {
    dev_ctx.template Alloc<T>(t2_grad);
    auto eigen_dt2 = phi::EigenTensor<T, 1>::From(*t2_grad, dim);
    eigen_dt2.device(place) =
        (eigen_dout.template cast<MPType>() * eigen_t1.template cast<MPType>() *
         static_cast<MPType>(value.to<T>()))
            .template cast<T>();
  }
}

template <typename T, typename Context>
void AddcmulGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& tensor1,
                       const DenseTensor& tensor2,
                       const DenseTensor& out_grad,
                       const Scalar& value,
                       DenseTensor* input_grad,
                       DenseTensor* tensor1_grad,
                       DenseTensor* tensor2_grad) {
  if (out_grad.numel() == 0) {
    if (input_grad)
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(::common::vectorize(input_grad->dims())),
          0,
          input_grad);
    if (tensor1_grad)
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(::common::vectorize(tensor1_grad->dims())),
          0,
          tensor1_grad);
    if (tensor2_grad)
      phi::Full<T, Context>(
          dev_ctx,
          phi::IntArray(::common::vectorize(tensor2_grad->dims())),
          0,
          tensor2_grad);
    return;
  }

  int rank = out_grad.dims().size();

  switch (rank) {
    case 0:
      AddcmulGradFunctionZero<Context, T>(dev_ctx,
                                          input,
                                          tensor1,
                                          tensor2,
                                          out_grad,
                                          value,
                                          input_grad,
                                          tensor1_grad,
                                          tensor2_grad);
      break;
    case 1:
      AddcmulGradFunction<Context, T, 1>(dev_ctx,
                                         input,
                                         tensor1,
                                         tensor2,
                                         out_grad,
                                         value,
                                         input_grad,
                                         tensor1_grad,
                                         tensor2_grad);
      break;
    case 2:
      AddcmulGradFunction<Context, T, 2>(dev_ctx,
                                         input,
                                         tensor1,
                                         tensor2,
                                         out_grad,
                                         value,
                                         input_grad,
                                         tensor1_grad,
                                         tensor2_grad);
      break;
    case 3:
      AddcmulGradFunction<Context, T, 3>(dev_ctx,
                                         input,
                                         tensor1,
                                         tensor2,
                                         out_grad,
                                         value,
                                         input_grad,
                                         tensor1_grad,
                                         tensor2_grad);
      break;
    case 4:
      AddcmulGradFunction<Context, T, 4>(dev_ctx,
                                         input,
                                         tensor1,
                                         tensor2,
                                         out_grad,
                                         value,
                                         input_grad,
                                         tensor1_grad,
                                         tensor2_grad);
      break;
    case 5:
      AddcmulGradFunction<Context, T, 5>(dev_ctx,
                                         input,
                                         tensor1,
                                         tensor2,
                                         out_grad,
                                         value,
                                         input_grad,
                                         tensor1_grad,
                                         tensor2_grad);
      break;
    case 6:
      AddcmulGradFunction<Context, T, 6>(dev_ctx,
                                         input,
                                         tensor1,
                                         tensor2,
                                         out_grad,
                                         value,
                                         input_grad,
                                         tensor1_grad,
                                         tensor2_grad);
      break;
    default:
      PADDLE_THROW(::common::errors::InvalidArgument(
          "AddcmulGrad only supports rank <= 6"));
  }
}

}  // namespace phi
