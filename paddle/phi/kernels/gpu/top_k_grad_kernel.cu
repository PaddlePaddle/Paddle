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
void TopkGradKernel(const Context& dev_ctx,
                    const DenseTensor& out_grad,
                    const DenseTensor& x,
                    const DenseTensor& indices,
                    int k,
                    int axis,
                    bool largest,
                    bool sorted,
                    DenseTensor* x_grad) {
  const auto& in_dims = x.dims();
  const auto& out_dims = indices.dims();

  // get the real the axis and the k
  if (axis < 0) {
    axis += in_dims.size();
  }
  const int& raw_height = in_dims[axis];

  // allocate the cuda memory for the x_grad
  T* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
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
  int block_size = ComputeBlockSize(post * k);
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
  int grid_size = std::min(max_blocks, pre);

  // lanuch the cuda kernel to assign the grad
  ops::AssignGradWithAxis<
      T><<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
      out_grad_data, indices_data, x_grad_data, pre, post, n, k);
}

}  // namespace phi

PD_REGISTER_KERNEL(top_k_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::TopkGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
