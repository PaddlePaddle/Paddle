/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/weight_only_linear_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/funcs/weight_dequant_functor.h"
#include "paddle/phi/kernels/fusion/cutlass/utils/cuda_utils.h"
#endif

namespace phi {

template <typename T, typename Context>
void WeightOnlyLinearGradKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& weight,
                                const paddle::optional<DenseTensor>& bias,
                                const DenseTensor& weight_scale,
                                const DenseTensor& out_grad,
                                const std::string& weight_dtype,
                                const int32_t arch,
                                DenseTensor* x_grad) {
#if defined(PADDLE_WITH_CUTLASS)
  int32_t cuda_arch_version;
  if (arch == 0) {
    // Note(Zhengzekang): user do not set the arch, we will get SM Arch from
    // device.
    cuda_arch_version = getSMVersion();
  } else {
    cuda_arch_version = arch;
  }
  PADDLE_ENFORCE_EQ(
      cuda_arch_version,
      80,
      phi::errors::InvalidArgument(
          "Currently weightonly linear grad only support arch = 80. "));

  int n = weight_scale.dims()[0];
  int k = weight.dims()[1];
  dev_ctx.template Alloc<T>(x_grad);
  DenseTensor weight_dequantized;
  weight_dequantized.Resize({{n, k}});
  dev_ctx.template Alloc<T>(&weight_dequantized);
  std::string algo =
      weight_dtype == "int8" ? "weight_only_int8" : "weight_only_int4";
  WeightDequantize<T, Context>(
      dev_ctx, weight, weight_scale, algo, true, &weight_dequantized);
  MatmulKernel<T, Context>(
      dev_ctx, out_grad, weight_dequantized, false, false, x_grad);
#else
  PADDLE_THROW(
      phi::errors::PreconditionNotMet("Not compiled with WITH_CUTLASS=ON"));
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(weight_only_linear_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyLinearGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
