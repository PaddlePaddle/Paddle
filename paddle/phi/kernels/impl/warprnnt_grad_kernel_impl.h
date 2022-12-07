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
#include "paddle/phi/kernels/funcs/warp_transducer/include/rnnt.h"

namespace phi {

template <typename T, typename Context>
void WarprnntGradKernel(const Context& dev_ctx,
                        const DenseTensor& logits,
                        const DenseTensor& logits_length,
                        const DenseTensor& warprnntgrad,
                        const DenseTensor& loss_grad,
                        int blank,
                        float fastemit_lambda,
                        int num_threads,
                        DenseTensor* logits_grad) {
  dev_ctx.template Alloc<T>(logits_grad);

  std::cout << "loss_grad: " << loss_grad << std::endl;
  std::cout << "warprnntgrad: " << warprnntgrad << std::endl;
  std::cout << "logits_grad: " << *logits_grad << std::endl;

  int B = warprnntgrad.dims()[0];     // B
  int Tmax = warprnntgrad.dims()[1];  // Tmax
  int Umax = warprnntgrad.dims()[2];  // Umax
  int D = warprnntgrad.dims()[3];     // D

  // (B,)
  auto loss_grad_e = EigenTensor<T, 1>::From(loss_grad);
  // std::cout << "loss_grad_e: " << loss_grad_e.eval() << std::endl;
  // (B, T, U, D)
  auto warprnntgrad_e = EigenTensor<T, 4>::From(warprnntgrad);
  // std::cout << "warprnntgrad_e: " << warprnntgrad_e.eval() << std::endl;

  auto logits_grad_e = EigenTensor<T, 4>::From(*logits_grad);

  Eigen::DSizes<int, 4> grad_shape(B, 1, 1, 1);
  Eigen::DSizes<int, 4> bcast(1, Tmax, Umax, D);
  auto logits_g =
      warprnntgrad_e * loss_grad_e.reshape(grad_shape).broadcast(bcast).eval();

  auto* place = dev_ctx.eigen_device();
  logits_grad_e.device(*place) = logits_g;

  // std::cout << "logits_g: " << logits_g.eval() << std::endl;
}

}  // namespace phi
