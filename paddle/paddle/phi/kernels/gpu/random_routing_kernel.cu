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

#include "paddle/phi/kernels/random_routing_kernel.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

#define CEIL(_x_, _y_) (((_x_)-1) / (_y_) + 1)
#define PERTHREAD_EXPERTS 256
#define WARP_SIZE 32

const int CUDA_NUM_THREADS = 512;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void random_routing_kernel(int64_t* data,
                                      const int64_t length,
                                      const size_t N,
                                      const size_t D,
                                      const T* prob,
                                      const int64_t* topk_idx,
                                      const T* topk_value) {
  CUDA_KERNEL_LOOP(idx, length) {
    size_t row = idx / D;
    size_t col = idx % D;
    if (col != 1) return;
    if (static_cast<T>(2) * topk_value[idx] < prob[row]) {
      data[idx] = static_cast<int64_t>(-1);
    }
  }
}

template <typename T, typename Context>
void RandomRoutingKernel(const Context& dev_ctx,
                         const DenseTensor& prob,
                         const DenseTensor& topk_value,
                         const DenseTensor& topk_idx,
                         DenseTensor* out) {
  phi::Copy(dev_ctx, topk_idx, dev_ctx.GetPlace(), false, out);

  size_t N = topk_idx.dims()[0];
  size_t D = topk_idx.dims()[1];

  int64_t num_idx = topk_idx.numel();

  auto prob_data = prob.data<T>();
  auto topk_value_data = topk_value.data<T>();
  auto topk_idx_data = topk_idx.data<int64_t>();
  auto out_data = out->data<int64_t>();

  random_routing_kernel<T>
      <<<GET_BLOCKS(num_idx), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          out_data, num_idx, N, D, prob_data, topk_idx_data, topk_value_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(random_routing,
                   GPU,
                   ALL_LAYOUT,
                   phi::RandomRoutingKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
