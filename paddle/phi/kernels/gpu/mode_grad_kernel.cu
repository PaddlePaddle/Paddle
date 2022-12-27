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

#include "paddle/phi/kernels/mode_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/mode.h"

namespace phi {

template <typename T>
__global__ void AssignGradWithAxis(const T* grad_out,
                                   const int64_t* indices,
                                   T* grad_in,
                                   int pre,
                                   int post,
                                   int raw_height,
                                   int k) {
  // raw_height is the length of topk axis
  for (int i = blockIdx.x; i < pre; i += gridDim.x) {
    int base_index = i * post * k;
    int base_grad = i * post * raw_height;
    for (int j = threadIdx.x; j < raw_height * post; j += blockDim.x) {
      grad_in[base_grad + j] = static_cast<T>(0);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < k * post; j += blockDim.x) {
      int64_t idx_ij = indices[base_index + j];
      int64_t in_ij = base_grad + (idx_ij * post) + (j % post);
      grad_in[in_ij] = grad_out[base_index + j];
    }
  }
}

template <typename T, typename Context>
void ModeGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& indices,
                    const DenseTensor& out_grad,
                    int axis,
                    bool keepdim,
                    DenseTensor* x_grad) {
  const auto& in_dims = x.dims();
  auto out_dims = indices.dims();

  if (axis < 0) axis += in_dims.size();
  // allocate the cuda memory for the x_grad
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* out_grad_data = out_grad.data<T>();
  const int64_t* indices_data = indices.data<int64_t>();

  int pre, n, post;
  funcs::GetDims(in_dims, axis, &pre, &n, &post);

  // calcluate the block and grid num
  int block_size = funcs::ComputeBlockSize(post);
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
  int grid_size = std::min(max_blocks, pre);
  AssignGradWithAxis<T><<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
      out_grad_data, indices_data, x_grad_data, pre, post, n, 1);
}

}  // namespace phi

PD_REGISTER_KERNEL(mode_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ModeGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
