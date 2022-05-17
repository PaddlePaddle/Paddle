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

#include "paddle/phi/kernels/asgd_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void ASGDKernel(const T* param,
                           const T* grad,
                           const T* learning_rate,
                           const T* avg_param,
                           const T* current_step,
                           float t0,
                           size_t num,
                           T* param_out,
                           T* avg_param_out) {
  T lr = learning_rate[0];
  CUDA_KERNEL_LOOP(i, num) { param_out[i] = param[i] - lr * grad[i]; }
  T current_step_data = current_step[0];
  if (current_step_data <= t0) {
    memcpy(avg_param_out, param, num * sizeof(T));
  } else {
    const auto mu1 = 1 / (current_step_data - t0);
    const auto mu2 = 1 - mu1;
    CUDA_KERNEL_LOOP(i, num) {
      avg_param_out[i] = mu2 * avg_param[i] + mu1 * param_out[i];
    }
  }
}

template <typename T>
__global__ void IncreaseStep(const T* step, T* step_out) {
  *step_out = *step + 1;
}

template <typename T, typename Context>
void AsgdKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const DenseTensor& learning_rate,
                const DenseTensor& grad,
                const DenseTensor& avg_param,
                const DenseTensor& current_step,
                float t0,
                DenseTensor* param_out,
                DenseTensor* avg_param_out,
                DenseTensor* current_step_out) {
  int block = 512;
  int grid = (param.numel() + block - 1) / block;

  ASGDKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
      param.data<T>(),
      grad.data<T>(),
      learning_rate.data<T>(),
      avg_param.data<T>(),
      current_step.data<T>(),
      t0,
      param.numel(),
      param_out->mutable_data<T>(dev_ctx.GetPlace()),
      avg_param_out->mutable_data<T>(dev_ctx.GetPlace()));

  IncreaseStep<T><<<1, 1, 0, dev_ctx.stream()>>>(
      current_step.data<T>(),
      current_step_out->mutable_data<T>(dev_ctx.GetPlace()));
}

}  // namespace phi

PD_REGISTER_KERNEL(asgd, GPU, ALL_LAYOUT, phi::AsgdKernel, float, double) {}
