// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename MTP, int k>
__global__ void combine_no_weight_kernel(const T* __restrict__ x,
                                         const int* __restrict__ scatter_index,
                                         T* __restrict__ y,
                                         const int64_t hidden_size,
                                         const int64_t seqlen) {
  extern __shared__ char shared_mem[];
  int64_t* shared_indices = reinterpret_cast<int64_t*>(shared_mem);

  int64_t seq_i = blockIdx.x;
  for (int ki = threadIdx.x; ki < k; ki += blockDim.x) {
    shared_indices[ki] = scatter_index[seq_i * k + ki];
  }
  __syncthreads();
  for (int h_i = threadIdx.x; h_i < hidden_size; h_i += blockDim.x) {
    MTP sum = static_cast<MTP>(0);
#pragma unroll
    for (int ki = 0; ki < k; ++ki) {
      int64_t scatter_idx = shared_indices[ki];
      T x_val = x[scatter_idx * hidden_size + h_i];
      sum += static_cast<MTP>(x_val);
    }
    y[seq_i * hidden_size + h_i] = static_cast<T>(sum);
  }
}

template <typename T>
void moe_combine_no_weight_fwd(const T* x,
                               const int* scatter_index,
                               T* y,
                               const int64_t k,
                               const int64_t seqlen,
                               const int64_t hidden_size,
                               cudaStream_t stream) {
  int threads_per_block = 1024;
  dim3 blockDim(threads_per_block);
  dim3 gridDim(seqlen);
  size_t sharedMemSize = k * sizeof(int64_t);

#define CALL_KERNEL(K)                                 \
  case K:                                              \
    combine_no_weight_kernel<T, float, K>              \
        <<<gridDim, blockDim, sharedMemSize>>>(        \
            x, scatter_index, y, hidden_size, seqlen); \
    break;

  switch (k) {
    CALL_KERNEL(1);
    CALL_KERNEL(2);
    CALL_KERNEL(3);
    CALL_KERNEL(4);
    CALL_KERNEL(5);
    CALL_KERNEL(6);
    CALL_KERNEL(7);
    CALL_KERNEL(8);
    CALL_KERNEL(9);
    CALL_KERNEL(10);
    CALL_KERNEL(11);
    CALL_KERNEL(12);
    CALL_KERNEL(13);
    CALL_KERNEL(14);
    CALL_KERNEL(15);
    CALL_KERNEL(16);
    default:
      PADDLE_THROW(phi::errors::InvalidArgument("Invalid k value."));
      break;
  }
#undef CALL_KERNEL
}

template <typename T, typename Context>
void MoeCombineNoWeightKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& scatter_index,
                              DenseTensor* y) {
  const auto x_shape = x.dims();
  const int64_t hidden_size = x_shape[1];

  const auto scatter_index_shape = scatter_index.dims();
  const int64_t seqlen = scatter_index_shape[0];
  const int64_t k = scatter_index_shape[1];

  dev_ctx.template Alloc<T>(y);

  moe_combine_no_weight_fwd<T>(x.data<T>(),
                               scatter_index.data<int>(),
                               y->data<T>(),
                               k,
                               seqlen,
                               hidden_size,
                               dev_ctx.stream());
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_combine_no_weight,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeCombineNoWeightKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
