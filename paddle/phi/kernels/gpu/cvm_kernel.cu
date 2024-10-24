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
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/cvm_kernel_impl.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void CvmComputeKernel(const bool use_cvm,
                                 const int64_t item_width,
                                 const T* X,
                                 T* Y,
                                 int64_t numel) {
  CUDA_KERNEL_LOOP(i, numel) {
    if (use_cvm) {
      if (i % item_width == 0) {
        Y[i] = log(X[i] + 1);
      } else if (i % item_width == 1) {
        Y[i] = log(X[i] + 1) - log(X[i - 1] + 1);
      } else {
        Y[i] = X[i];
      }
    } else {
      Y[i] = X[i / (item_width - 2) * item_width + i % (item_width - 2) + 2];
    }
  }
}

template <typename T, typename Context>
void CVMCUDAKernel(const Context& dev_ctx,
                   const DenseTensor& x_in,
                   const DenseTensor& cvm,
                   bool use_cvm,
                   DenseTensor* out) {
  const auto* x = &x_in;
  const T* x_data = x->data<T>();

  auto batch_size = x->dims()[0];
  auto numel = x->numel();
  auto item_size = numel / batch_size;

  auto* y = out;
  T* y_data = dev_ctx.template Alloc<T>(y);

  // for Input X do not have Lod Information.
  auto stream = dev_ctx.stream();
  if (x->NumLevels() == 0) {
    CvmComputeKernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                           PADDLE_CUDA_NUM_THREADS,
                       PADDLE_CUDA_NUM_THREADS,
                       0,
                       stream>>>(
        use_cvm, item_size, x_data, y_data, y->numel());
  } else {
    auto lod = x->lod()[0];
    PADDLE_ENFORCE_EQ(
        batch_size,
        lod[lod.size() - 1],
        common::errors::PreconditionNotMet(
            "Input(X)'s dim[0] must be equal to last element of lod"));
    CvmComputeKernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                           PADDLE_CUDA_NUM_THREADS,
                       PADDLE_CUDA_NUM_THREADS,
                       0,
                       stream>>>(
        use_cvm, item_size, x_data, y_data, y->numel());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cvm, GPU, ALL_LAYOUT, phi::CVMCUDAKernel, float, double) {}
