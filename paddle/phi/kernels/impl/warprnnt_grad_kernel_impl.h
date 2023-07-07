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

#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void WarprnntGradKernel(const Context& dev_ctx,
                        const DenseTensor& input UNUSED,
                        const DenseTensor& input_lengths UNUSED,
                        const DenseTensor& warprnntgrad,
                        const DenseTensor& loss_grad,
                        int blank UNUSED,
                        float fastemit_lambda UNUSED,
                        DenseTensor* input_grad) {
  dev_ctx.template Alloc<T>(input_grad);

  int B = warprnntgrad.dims()[0];     // B
  int Tmax = warprnntgrad.dims()[1];  // Tmax
  int Umax = warprnntgrad.dims()[2];  // Umax
  int D = warprnntgrad.dims()[3];     // D

  // (B,)
  auto loss_grad_e = EigenTensor<T, 1>::From(loss_grad);

  // (B, T, U, D)
  auto warprnntgrad_e = EigenTensor<T, 4>::From(warprnntgrad);
  auto acts_grad_e = EigenTensor<T, 4>::From(*input_grad);

  Eigen::DSizes<int, 4> grad_shape(B, 1, 1, 1);
  Eigen::DSizes<int, 4> bcast(1, Tmax, Umax, D);
  auto acts_g =
      warprnntgrad_e * loss_grad_e.reshape(grad_shape).broadcast(bcast).eval();

  auto* place = dev_ctx.eigen_device();
  acts_grad_e.device(*place) = acts_g;
}

}  // namespace phi
