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

#include "paddle/phi/kernels/assign_pos_kernel.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__global__ void AssignPos(T* cum_count,
                          const T* numbers,
                          T* out,
                          int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    int number_idx = numbers[i];
    if (number_idx > -1) {
      int p = phi::CudaAtomicAdd(cum_count + number_idx, -1);
      out[p - 1] = i;
    }
  }
}

template <typename T, typename Context>
void AssignPosKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& cum_count,
                     const DenseTensor& eff_num_len,
                     DenseTensor* out) {
  // assign pos decides which tokens should be fetched belong to specially
  // counter orderingly.
  auto cum_count_ptr = &cum_count;      // (counter number) int32 | int64
  auto numbers = &x;                    // (batch_size * seq_len, topk) int32
  auto eff_num_len_ptr = &eff_num_len;  // (sum(cum_count))
  auto out_ptr = &out;                  // (cum_count) value ranges
                                        // from 0 to batch_size *
                                        // seq_len * topk
  auto numel = numbers->numel();
  T* cum_data = const_cast<T*>(cum_count_ptr->data<T>());
  auto cum_size = cum_count_ptr->numel();

  phi::DenseTensor cpu_eff_num_len;
  int64_t cpu_eff_num_len_data = 0;
  bool is_cpu_place = eff_num_len_ptr->place() == phi::CPUPlace();
  if (is_cpu_place) {
    cpu_eff_num_len_data = eff_num_len_ptr->data<T>()[0];
  } else {
    phi::Copy(dev_ctx, eff_num_len, phi::CPUPlace(), false, &cpu_eff_num_len);
    cpu_eff_num_len_data = cpu_eff_num_len.data<T>()[0];
  }

  phi::DDim out_dims = common::make_ddim({cpu_eff_num_len_data});
  out->Resize(out_dims);
  auto out_data = dev_ctx.template Alloc<T>(out);

  const T* num_data = numbers->data<T>();

  int blocks = NumBlocks(numel);
  int threads = kNumCUDAThreads;

  AssignPos<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
      cum_data, num_data, out_data, numel);
}

}  // namespace phi

PD_REGISTER_KERNEL(assign_pos, GPU, ALL_LAYOUT, phi::AssignPosKernel, int64_t) {
}
