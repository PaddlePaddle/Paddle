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

#include "paddle/phi/kernels/kthvalue_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/top_k_function_cuda.h"

namespace phi {
static int getBlockSize(int col) {
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
}

template <typename T, typename Context>
void KthvalueGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& indices,
                        const DenseTensor& d_out,
                        int k,
                        int axis,
                        bool keepdim,
                        DenseTensor* d_x) {
  const auto& in_dims = x.dims();
  auto out_dims = indices.dims();
  T* x_grad_data = dev_ctx.template Alloc<T>(d_x);
  // For 0D Tensor
  if (in_dims.size() == 0) {
    phi::funcs::set_constant(dev_ctx, d_x, static_cast<T>(1.0));
    return;
  }

  if (axis < 0) axis += in_dims.size();

  const T* out_grad_data = d_out.data<T>();
  const int64_t* indices_data = indices.data<int64_t>();
  int pre, n, post;
  phi::funcs::GetDims(in_dims, axis, &pre, &n, &post);
  int block_size = getBlockSize(post * k);
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
  int grid_size = std::min(max_blocks, pre);
  phi::funcs::AssignGradWithAxis<T>
      <<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
          out_grad_data, indices_data, x_grad_data, pre, post, n, 1);
}

}  // namespace phi

PD_REGISTER_KERNEL(kthvalue_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::KthvalueGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
