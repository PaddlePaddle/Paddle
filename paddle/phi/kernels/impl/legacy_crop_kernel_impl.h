// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/strided_memcpy.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename Context, typename T, size_t D>
void CropFunction(const Context &dev_ctx,
                  const DenseTensor &input_x,
                  const IntArray &offsets_in,
                  DenseTensor *out) {
  auto *x = &input_x;
  auto out_dims = out->dims();
  if (out_dims[0] == -1) {
    out_dims[0] = x->dims()[0];
  }
  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  auto offsets = offsets_in.GetData();

  auto x_tensor = EigenTensor<T, D>::From(*x);
  auto out_tensor = EigenTensor<T, D>::From(*out);
  Eigen::DSizes<Eigen::DenseIndex, D> e_offsets;
  Eigen::DSizes<Eigen::DenseIndex, D> e_shape;
  for (size_t i = 0; i < D; ++i) {
    e_offsets[i] = offsets[i];
    e_shape[i] = out->dims()[i];
  }
  auto &place = *dev_ctx.eigen_device();
  phi::funcs::EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_tensor, x_tensor, e_offsets, e_shape);
}

template <typename T, typename Context>
void CropKernel(const Context &dev_ctx,
                const DenseTensor &x,
                const paddle::optional<DenseTensor> &y,
                const IntArray &offsets_in,
                const std::vector<int> &shape,
                DenseTensor *out) {
  int rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1,
      common::errors::InvalidArgument(
          "The number of dimensions of the Input(X) for CropOp must be "
          "greater than or equal to 1, but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      common::errors::InvalidArgument(
          "The number of dimensions of the Input(X) for CropOp must be "
          "less than or equal to 6, but the value received is %d.",
          rank));
  switch (rank) {
    case 1:
      CropFunction<Context, T, 1>(dev_ctx, x, offsets_in, out);
      break;
    case 2:
      CropFunction<Context, T, 2>(dev_ctx, x, offsets_in, out);
      break;
    case 3:
      CropFunction<Context, T, 3>(dev_ctx, x, offsets_in, out);
      break;
    case 4:
      CropFunction<Context, T, 4>(dev_ctx, x, offsets_in, out);
      break;
    case 5:
      CropFunction<Context, T, 5>(dev_ctx, x, offsets_in, out);
      break;
    case 6:
      CropFunction<Context, T, 6>(dev_ctx, x, offsets_in, out);
      break;
  }
}

template <typename Context, typename T, size_t D>
void CropGradFunction(const Context &dev_ctx,
                      const DenseTensor &input_x,
                      const DenseTensor &out_grad,
                      const IntArray &offsets_in,
                      DenseTensor *x_grad) {
  auto *d_x = x_grad;
  if (d_x != nullptr) {
    auto *d_out = &out_grad;
    dev_ctx.template Alloc<T>(d_x);
    auto offsets = offsets_in.GetData();
    std::array<std::pair<int64_t, int64_t>, D> paddings;
    for (size_t i = 0; i < D; ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = d_x->dims()[i] - d_out->dims()[i] - offsets[i];
    }
    auto d_x_tensor = EigenTensor<T, D>::From(*d_x);
    auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
    auto &place = *dev_ctx.eigen_device();
    phi::funcs::EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
        place, d_x_tensor, d_out_tensor, paddings, static_cast<T>(0));
  }
}

template <typename T, typename Context>
void CropGradKernel(const Context &dev_ctx,
                    const DenseTensor &x,
                    const paddle::optional<DenseTensor> &y,
                    const DenseTensor &out_grad,
                    const IntArray &offsets,
                    const std::vector<int> &shape,
                    DenseTensor *x_grad) {
  size_t rank = out_grad.dims().size();
  PADDLE_ENFORCE_GE(rank,
                    1,
                    common::errors::InvalidArgument(
                        "The number of dimensions of the input 'Out@GRAD' for "
                        "CropGrad must be greater than or equal "
                        "to 1, but the value received is %d.",
                        rank));
  PADDLE_ENFORCE_LE(rank,
                    6,
                    common::errors::InvalidArgument(
                        "The number of dimensions of the input 'Out@GRAD' for "
                        "CropGrad must be less than or equal "
                        "to 6, but the value received is %d.",
                        rank));
  switch (rank) {
    case 1:
      CropGradFunction<Context, T, 1>(dev_ctx, x, out_grad, offsets, x_grad);
      break;
    case 2:
      CropGradFunction<Context, T, 2>(dev_ctx, x, out_grad, offsets, x_grad);
      break;
    case 3:
      CropGradFunction<Context, T, 3>(dev_ctx, x, out_grad, offsets, x_grad);
      break;
    case 4:
      CropGradFunction<Context, T, 4>(dev_ctx, x, out_grad, offsets, x_grad);
      break;
    case 5:
      CropGradFunction<Context, T, 5>(dev_ctx, x, out_grad, offsets, x_grad);
      break;
    case 6:
      CropGradFunction<Context, T, 6>(dev_ctx, x, out_grad, offsets, x_grad);
      break;
  }
}
}  // namespace phi
