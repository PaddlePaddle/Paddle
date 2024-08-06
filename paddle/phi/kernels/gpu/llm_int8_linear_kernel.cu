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

#include "paddle/phi/kernels/llm_int8_linear_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
#include "paddle/phi/kernels/impl/llm_int8_matmul_kernel_impl.h"
#endif

namespace phi {

template <typename T, typename Context>
void llm_int8_compute(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& weight,
                      const paddle::optional<DenseTensor>& bias,
                      const DenseTensor& weight_scale,
                      const float threshold,
                      DenseTensor* out) {
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11020
  DenseTensor cublaslt_workspace;
  cublaslt_workspace.Resize({{3000000}});
  dev_ctx.template Alloc<int8_t>(&cublaslt_workspace);
  const auto x_dims = x.dims();
  const auto w_dims = weight.dims();
  int k = w_dims[1];
  int n = w_dims[0];
  int m = x.numel() / k;
  // mk * transpose(nk) = mn
  llm_int8::LLMGemm<T>(dev_ctx,
                       &weight,
                       &x,
                       &weight_scale,
                       threshold,
                       out,
                       &cublaslt_workspace,
                       "llm_int8_mat_mul",
                       m,
                       k,
                       n);
  if (bias) {
    phi::AddKernel<T, Context>(dev_ctx, *out, bias.get(), out);
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "llm_int8_linear op needs paddle with cuda and cuda version >= 11.2"));
#endif
}

template <typename T, typename Context>
void LLMInt8LinearKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& weight,
                         const paddle::optional<DenseTensor>& bias,
                         const DenseTensor& weight_scale,
                         const float threshold,
                         DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  llm_int8_compute<T, Context>(
      dev_ctx, x, weight, bias, weight_scale, threshold, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(llm_int8_linear,
                   GPU,
                   ALL_LAYOUT,
                   phi::LLMInt8LinearKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
