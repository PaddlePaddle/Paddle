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
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/cvm_kernel_impl.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void CvmGradComputeKernel(const bool use_cvm,
                                     const int64_t item_width,
                                     const T* CVM,
                                     const T* DY,
                                     T* DX,
                                     bool has_lod,
                                     const size_t* lod,
                                     size_t lod_size,
                                     int64_t numel) {
  CUDA_KERNEL_LOOP(i, numel) {
    int offset = i % item_width;
    if (offset <= 1) {
      int cvm_id = i / item_width;
      if (has_lod) {
        int low = 1;
        int high = lod_size - 1;
        while (low < high) {
          int mid = (low + high) / 2;
          if (cvm_id < lod[mid])
            high = mid;
          else
            low = mid + 1;
        }
        cvm_id = low - 1;
      }
      DX[i] = CVM[2 * cvm_id + offset];
    } else {
      if (use_cvm) {
        DX[i] = DY[i];
      } else {
        DX[i] = DY[i / item_width * (item_width - 2) + i % item_width - 2];
      }
    }
  }
}

template <typename T, typename Context>
void CVMGradCUDAKernel(const Context& dev_ctx,
                       const DenseTensor& x_in,
                       const DenseTensor& cvm_in,
                       const DenseTensor& out_grad,
                       bool use_cvm,
                       DenseTensor* x_grad) {
  auto* dx = x_grad;
  T* dx_data = dev_ctx.template Alloc<T>(dx);

  const phi::DenseTensor* cvm = &cvm_in;
  const T* cvm_data = cvm->data<T>();

  const auto* dOut = &out_grad;
  const T* dout_data = dOut->data<T>();

  auto offset = 2;
  auto batch_size = dx->dims()[0];
  auto dx_numel = dx->numel();
  auto item_size = dx_numel / batch_size;

  // for Input X do not have Lod Information.
  auto stream = dev_ctx.stream();
  if (dx->NumLevels() == 0) {
    CvmGradComputeKernel<<<(dx_numel + PADDLE_CUDA_NUM_THREADS - 1) /
                               PADDLE_CUDA_NUM_THREADS,
                           PADDLE_CUDA_NUM_THREADS,
                           0,
                           stream>>>(use_cvm,
                                     item_size,
                                     cvm_data,
                                     dout_data,
                                     dx_data,
                                     false,
                                     NULL,
                                     0,
                                     dx_numel);
  } else {
    auto lod = dx->lod()[0];
    PADDLE_ENFORCE_EQ(
        batch_size,
        lod[lod.size() - 1],
        common::errors::PreconditionNotMet(
            "Output(X@GRAD)'s dim[0] must be equal to last element of lod"));
    phi::MixVector<size_t> mixv_lod(&lod);
    CvmGradComputeKernel<<<(dx_numel + PADDLE_CUDA_NUM_THREADS - 1) /
                               PADDLE_CUDA_NUM_THREADS,
                           PADDLE_CUDA_NUM_THREADS,
                           0,
                           stream>>>(use_cvm,
                                     item_size,
                                     cvm_data,
                                     dout_data,
                                     dx_data,
                                     true,
                                     mixv_lod.CUDAData(dev_ctx.GetPlace()),
                                     lod.size(),
                                     dx_numel);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    cvm_grad, GPU, ALL_LAYOUT, phi::CVMGradCUDAKernel, float, double) {}
