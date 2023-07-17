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

#include "paddle/phi/kernels/attn_matmul_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"

namespace phi {
template <typename T, typename Context>
void AttnMatmulKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& weight,
                      const paddle::optional<DenseTensor>& bias,
                      const bool transpose_weight,
                      DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  int k = x_dims[x_dims.size() - 1];
  int m = x.numel() / k;
  int n = weight.dims()[transpose_weight ? 0 : 1];

  auto attn_matmul_compute = fusion::AttnMatMul<T>(
      dev_ctx, false, transpose_weight, m, n, k, bias ? true : false);
  attn_matmul_compute.ComputeForward(
      &weight, &x, bias ? &(bias.get()) : nullptr, out, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(attn_matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::AttnMatmulKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
