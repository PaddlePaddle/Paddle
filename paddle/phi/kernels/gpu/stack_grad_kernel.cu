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

#include "paddle/phi/kernels/stack_grad_kernel.h"

#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename IntType>
__global__ void UnStackHelperCUDAKernel(const T* __restrict__ input,
                                        int pre_dim_size,
                                        int split_dim_size,
                                        int suf_dim_size,
                                        int num_split,
                                        T** output_ptrs) {
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  // In this case they are equal
  assert(split_dim_size % num_split == 0);

  IntType size = pre_dim_size * split_dim_size * suf_dim_size;
  IntType each_dim_size = split_dim_size / num_split;

  for (IntType offset = blockIdx.x * blockDim.x + threadIdx.x; offset < size;
       offset += blockDim.x * gridDim.x) {
    IntType i = offset / (split_dim_size * suf_dim_size);
    IntType j = (offset % (split_dim_size * suf_dim_size)) / suf_dim_size;
    IntType k = offset % suf_dim_size;

    T* output = output_ptrs[j / each_dim_size];
    if (output == nullptr) {
      return;
    }
    IntType output_ind = i * each_dim_size * suf_dim_size +
                         (j % each_dim_size) * suf_dim_size + k;
    *(output + output_ind) = input[offset];
  }
}

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const DenseTensor& out,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  if (axis < 0) axis += out.dims().size();

  int n = out.dims()[axis];
  PADDLE_ENFORCE_EQ(n,
                    x_grad.size(),
                    phi::errors::InvalidArgument(
                        "Output x_grad size should be equal to n, but"
                        " received n is:%d x_grad size is:%d.",
                        n,
                        x_grad.size()));

  // x_grad is output, so save each data address, then copy each dy into dx_data
  std::vector<T*> outputs(n);
  for (size_t j = 0; j < x_grad.size(); ++j) {
    if (x_grad[j] == nullptr) {
      outputs[j] = nullptr;
      continue;
    }
    if (x_grad[j]->numel() != 0UL) {
      T* ptr = dev_ctx.template Alloc<T>(x_grad[j]);
      outputs[j] = ptr;
    } else {
      outputs[j] = nullptr;
    }
  }
  auto dy_data = out.data<T>();
  // each x_grad should have same shape
  int dy_pre = 1, dy_suf = 1;
  auto dy_dims = out.dims();
  int split_dim = n;
  for (int i = 0; i < axis; ++i) {
    dy_pre *= dy_dims[i];
  }
  dy_suf = out.numel() / (split_dim * dy_pre);

  auto tmp_out_data =
      paddle::memory::Alloc(dev_ctx, outputs.size() * sizeof(T*));
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       tmp_out_data->ptr(),
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(outputs.data()),
                       outputs.size() * sizeof(T*),
                       dev_ctx.stream());

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, dy_pre * split_dim * dy_suf);

  if (out.numel() < std::numeric_limits<int32_t>::max()) {
    UnStackHelperCUDAKernel<T, int32_t><<<config.block_per_grid.x,
                                          config.thread_per_block.x,
                                          0,
                                          dev_ctx.stream()>>>(
        dy_data,
        dy_pre,
        split_dim,
        dy_suf,
        split_dim,
        reinterpret_cast<T**>(tmp_out_data->ptr()));
  } else {
    UnStackHelperCUDAKernel<T, int64_t><<<config.block_per_grid.x,
                                          config.thread_per_block.x,
                                          0,
                                          dev_ctx.stream()>>>(
        dy_data,
        dy_pre,
        split_dim,
        dy_suf,
        split_dim,
        reinterpret_cast<T**>(tmp_out_data->ptr()));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::StackGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
