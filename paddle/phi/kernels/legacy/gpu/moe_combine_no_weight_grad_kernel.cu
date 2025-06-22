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
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {

template <typename T, typename MTP, int VecSize>
__global__ void combine_no_weight_bwd_kernel(const int* scatter_index,
                                             const T* grad_y,
                                             T* grad_x,
                                             const int64_t k,
                                             const int64_t seqlen,
                                             const int64_t hidden_size) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT grad_y_vec;
  int i = blockIdx.x;   // Batch index (sequence length)
  int ki = blockIdx.y;  // Sequence index

  if (i < seqlen && ki < k) {
    int idx = scatter_index[i * k + ki];  // Index into x

    // Loop over h dimension in strides of block
    for (int h_i = threadIdx.x * VecSize; h_i < hidden_size;
         h_i += blockDim.x * VecSize) {
      phi::Load<T, VecSize>(&(grad_y[i * hidden_size + h_i]), &grad_y_vec);
      phi::Store<T, VecSize>(grad_y_vec, &grad_x[idx * hidden_size + h_i]);
    }
  }
}

template <typename T>
void moe_combine_no_weight_bwd(const int* scatter_index,
                               const T* grad_y,
                               T* grad_x,
                               const int64_t k,
                               const int64_t seqlen,
                               const int64_t hidden_size,
                               cudaStream_t stream) {
  int block_size = 512;
  int grid_size_i = seqlen;
  int grid_size_k = k;
  dim3 blockDim(block_size);
  dim3 gridDim(grid_size_i, grid_size_k);

  constexpr int max_pack_size = 16 / sizeof(T);
  if (hidden_size % max_pack_size == 0) {
    combine_no_weight_bwd_kernel<T, float, max_pack_size>
        <<<gridDim, blockDim, 0, stream>>>(
            scatter_index, grad_y, grad_x, k, seqlen, hidden_size);
  } else {
    combine_no_weight_bwd_kernel<T, float, 1><<<gridDim, blockDim, 0, stream>>>(
        scatter_index, grad_y, grad_x, k, seqlen, hidden_size);
  }
}

template <typename T, typename Context>
void MoeCombineNoWeightGradKernel(const Context& dev_ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& scatter_index,
                                  const DenseTensor& grad_y,
                                  DenseTensor* grad_x) {
  const auto x_shape = x.dims();
  const int64_t hidden_size = x_shape[1];

  const auto scatter_index_shape = scatter_index.dims();
  const int64_t seqlen = scatter_index_shape[0];
  const int64_t k = scatter_index_shape[1];

  dev_ctx.template Alloc<T>(grad_x);
  phi::Full<T, Context>(
      dev_ctx, phi::IntArray(common::vectorize(grad_x->dims())), 0, grad_x);

  moe_combine_no_weight_bwd<T>(scatter_index.data<int>(),
                               grad_y.data<T>(),
                               grad_x->data<T>(),
                               k,
                               seqlen,
                               hidden_size,
                               dev_ctx.stream());
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_combine_no_weight_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeCombineNoWeightGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
