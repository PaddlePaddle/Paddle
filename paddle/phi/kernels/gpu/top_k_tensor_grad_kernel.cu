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

#include "paddle/phi/kernels/top_k_grad_kernel.h"

#include "paddle/fluid/operators/top_k_function_cuda.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

namespace ops = paddle::operators;

template <typename T, typename Context>
void TopKTensorGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& k_list,
                          const DenseTensor& indices,
                          const DenseTensor& out_grad,
                          int axis,
                          bool largest,
                          DenseTensor* x_grad) {
  const auto& in_dims = x.dims();
  const auto& out_dims = indices.dims();

  DenseTensor k_largest_tensor;
  phi::DDim k_largest_dim = phi::make_ddim({1});
  k_largest_tensor.Resize(k_largest_dim);
  dev_ctx.template Alloc<int>(&k_largest_tensor);
  int* k_largest_data = k_largest_tensor.data<int>();

  ops::getMaxK<int, 256><<<1, 256, 0, dev_ctx.stream()>>>(
      k_list.data<int>(), k_largest_data, k_list.numel());
  DenseTensor k_largest_host;
  phi::CPUPlace cpu;
  phi::Copy(dev_ctx, k_largest_tensor, cpu, false, &k_largest_host);
  int k_largest = k_largest_host.data<int>()[0];

  // get the real the axis and the k
  if (axis < 0) {
    axis += in_dims.size();
  }
  const int& raw_height = in_dims[axis];
  if (k_largest > raw_height) {
    k_largest = raw_height;
  }

  // allocate the cuda memory for the x_grad
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  ops::InitVal<T, 256>
      <<<1, 256, 0, dev_ctx.stream()>>>(x_grad_data, x_grad->numel());
  const T* out_grad_data = out_grad.data<T>();
  const int64_t* indices_data = indices.data<int64_t>();

  int pre, n, post;
  ops::GetDims(in_dims, axis, &pre, &n, &post);

  // calcluate the block and grid num
  auto ComputeBlockSize = [](int col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else if (col > 64 && col <= 128)
      return 128;
    else
      return 64;
  };
  int block_size = ComputeBlockSize(post * k_largest);
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
  int grid_size = std::min(max_blocks, pre);
  int bs_size = in_dims[0];
  int bs_offset = pre / bs_size;

  // lanuch the cuda kernel to assign the grad
  ops::AssignGradWithAxis<T>
      <<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
          out_grad_data,
          indices_data,
          x_grad_data,
          pre,
          post,
          n,
          k_largest,
          bs_offset,
          k_list.numel() > 1 ? k_list.data<int>() : nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(top_k_tensor_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopKTensorGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
