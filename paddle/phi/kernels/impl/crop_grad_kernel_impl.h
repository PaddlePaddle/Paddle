
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

#include "paddle/phi/kernels/crop_grad_kernel.h"

#include <vector>

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename Context, typename T, size_t D>
void CropTensorGradFunction(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out_grad,
                            const IntArray& offsets,
                            DenseTensor* x_grad) {
  if (x_grad != nullptr) {
    x_grad->Resize(x.dims());
    dev_ctx.template Alloc<T>(x_grad);

    auto offsets_vec = offsets.GetData();
    std::array<std::pair<int64_t, int64_t>, D> paddings;
    for (size_t i = 0; i < D; ++i) {
      paddings[i].first = offsets_vec[i];
      paddings[i].second =
          x_grad->dims()[i] - out_grad.dims()[i] - offsets_vec[i];
    }
    auto x_grad_tensor = EigenTensor<T, D>::From(*x_grad);
    auto out_grad_tensor = EigenTensor<T, D>::From(out_grad);
    auto& place = *dev_ctx.eigen_device();

    funcs::EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
        place, x_grad_tensor, out_grad_tensor, paddings, static_cast<T>(0));
  }
}

template <typename T, typename Context>
void CropGradKernel(const Context& dev_ctx,
                    const DenseTensor& out_grad,
                    const DenseTensor& x,
                    const IntArray& offsets,
                    DenseTensor* x_grad) {
  size_t rank = out_grad.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1,
      errors::InvalidArgument(
          "The number of dimensions of the input 'Out@GRAD' for "
          "Op(crop_tensor_grad) must be greater than or equal to 1, but the "
          "value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      6,
      errors::InvalidArgument(
          "The number of dimensions of the input 'Out@GRAD' for "
          "Op(crop_tensor_grad) must be less than or equal to 6, but the "
          "value received is %d.",
          rank));
  switch (rank) {
    case 1:
      CropTensorGradFunction<Context, T, 1>(
          dev_ctx, out_grad, x, offsets, x_grad);
      break;
    case 2:
      CropTensorGradFunction<Context, T, 2>(
          dev_ctx, out_grad, x, offsets, x_grad);
      break;
    case 3:
      CropTensorGradFunction<Context, T, 3>(
          dev_ctx, out_grad, x, offsets, x_grad);
      break;
    case 4:
      CropTensorGradFunction<Context, T, 4>(
          dev_ctx, out_grad, x, offsets, x_grad);
      break;
    case 5:
      CropTensorGradFunction<Context, T, 5>(
          dev_ctx, out_grad, x, offsets, x_grad);
      break;
    case 6:
      CropTensorGradFunction<Context, T, 6>(
          dev_ctx, out_grad, x, offsets, x_grad);
      break;
  }
}

}  // namespace phi
