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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
template <typename T, typename Context>
void SquaredL2NormGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& dout,
                             DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  PADDLE_ENFORCE_EQ(
      dout.numel(),
      1,
      phi::errors::InvalidArgument(
          "Input(GRAD@Out) of SquaredL2NormGradOP should be a scalar."));

  auto input = phi::EigenVector<T>::Flatten(x);
  auto d_out = phi::EigenVector<T>::Flatten(dout);
  auto d_x = phi::EigenVector<T>::Flatten(*dx);
  auto* place = dev_ctx.eigen_device();
  Eigen::DSizes<int, 1> x_dsize(x.numel());
  d_x.device(*place) = (d_out.broadcast(x_dsize) * input) * static_cast<T>(2.0);
}
}  // namespace phi
