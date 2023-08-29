// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/limit_by_capacity_kernel.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T>
__global__ void limit_by_capacity_impl(
    const T* expc, T* cap, T* out, const int n_expert, const int n_worker) {
  int eid, wid;
  CUDA_KERNEL_LOOP(i, (n_expert * n_worker)) {
    wid = i / n_expert;
    eid = i % n_expert;
    auto proposal = expc[wid * n_expert + eid];
    auto cap_left = phi::CudaAtomicAdd(cap + eid, proposal * (-1));
    if (cap_left >= proposal) {
      out[wid * n_expert + eid] = proposal;
    } else if (cap_left >= 0) {
      out[wid * n_expert + eid] = cap_left;
    } else {
      out[wid * n_expert + eid] = 0;
    }
  }
}

template <typename T, typename Context>
void LimitByCapacityKernel(const Context& dev_ctx,
                           const DenseTensor& expert_count,
                           const DenseTensor& capacity,
                           int n_worker,
                           DenseTensor* Out) {
  auto expert_count_ptr = &expert_count;
  auto n_expert = expert_count_ptr->numel() / n_worker;

  dim3 grid_dim(256);
  dim3 block_dim(1024);
  auto out_data = dev_ctx.template Alloc<T>(Out);
  const T* ec_data = expert_count_ptr->data<T>();

  phi::DenseTensor capacity_copy;
  phi::Copy(dev_ctx, capacity, dev_ctx.GetPlace(), false, &capacity_copy);
  T* cap_data = dev_ctx.template Alloc<T>(&capacity_copy);

  limit_by_capacity_impl<T><<<grid_dim, block_dim, 0, dev_ctx.stream()>>>(
      ec_data, cap_data, out_data, n_expert, n_worker);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    limit_by_capacity, GPU, ALL_LAYOUT, phi::LimitByCapacityKernel, int64_t) {}
