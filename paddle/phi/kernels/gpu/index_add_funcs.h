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

#pragma once
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/index_add_kernel.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void index_add_cuda_kernel(const T* x_ptr,
                                       T* output,
                                       T* add_grad,
                                       const int64_t* index,
                                       int64_t N,
                                       int64_t stride,
                                       int64_t size,
                                       int64_t delta,
                                       T add_val) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    int64_t pre_idx = idx / (stride * size);
    int64_t dim_idx = idx % (stride * size) / stride;
    int64_t src_dim_idx = index[dim_idx];
    int64_t output_idx =
        idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
    if (add_grad) add_grad[output_idx] = x_ptr[output_idx];
    if (output) output[output_idx] = add_val;
  }
}

template <typename T, typename Context>
void IndexAddBaseKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const IntArray& index_arr,
                         const Scalar& axis_scalar,
                         float add_value,
                         DenseTensor* output,
                         DenseTensor* add_grad) {
  if (output) phi::Copy(dev_ctx, input, dev_ctx.GetPlace(), false, output);

  auto axis = axis_scalar.to<int>();
  auto index_list = index_arr.GetData();
  int64_t index_size = static_cast<int64_t>(index_list.size());

  DenseTensor index;
  index.Resize(make_ddim({index_size}));
  int64_t* index_data = dev_ctx.template Alloc<int64_t>(&index);
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       index_data,
                       phi::CPUPlace(),
                       index_list.data(),
                       index_size * sizeof(int64_t),
                       dev_ctx.stream());

  auto output_dim = input.dims();
  axis = axis >= 0 ? axis : axis + output_dim.size();
  auto stride_dim = phi::stride(output_dim);
  int64_t stride = stride_dim[axis];
  int64_t size = index.dims()[0];
  int64_t delta = output_dim[axis] - size;

  auto* out_data = output ? output->data<T>() : nullptr;
  auto* add_grad_data = add_grad ? add_grad->data<T>() : nullptr;
  output_dim[axis] = size;
  int64_t numel = phi::product(output_dim);
  if (numel == 0) {
    return;
  }
  auto stream = dev_ctx.stream();

  unsigned int block_dim = PADDLE_CUDA_NUM_THREADS;
  dim3 grid_dim = dim3((numel + block_dim - 1) / block_dim);
  paddle::platform::LimitGridDim(dev_ctx, &grid_dim);

  T add_val = static_cast<T>(add_value);
  const T* x_ptr = input.data<T>();
  index_add_cuda_kernel<T><<<grid_dim, block_dim, 0, stream>>>(x_ptr,
                                                                out_data,
                                                                add_grad_data,
                                                                index_data,
                                                                numel,
                                                                stride,
                                                                size,
                                                                delta,
                                                                add_val);
}

}  // namespace phi