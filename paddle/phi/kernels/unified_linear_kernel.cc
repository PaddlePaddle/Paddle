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

#include "paddle/phi/kernels/unified_linear_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/elementwise_function.h"
#include "paddle/phi/kernels/unified_linear_utils.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/kernels/gpu/unified_linear_cuda.h"
#endif

namespace phi {

// Main unified linear kernel implementation
template <typename T, typename Context>
void UnifiedLinearKernel(const Context& dev_ctx,
                         const DenseTensor& A,
                         const DenseTensor& B,
                         const DenseTensor& C,
                         const DenseTensor& D,
                         const paddle::optional<DenseTensor>& Bias,
                         const paddle::optional<DenseTensor>& A_scale,
                         const paddle::optional<DenseTensor>& B_scale,
                         const paddle::optional<DenseTensor>& C_scale,
                         const paddle::optional<DenseTensor>& D_scale,
                         float alpha,
                         float beta,
                         bool trans_A,
                         bool trans_B,
                         bool trans_C,
                         const std::string& activation,
                         DenseTensor* out_D,
                         DenseTensor* out_D_scale) {
  // Convert string to activation type
  unified_linear::ActivationType activation_type =
      unified_linear::StringToActivationType(activation);

  // Determine operation type based on tensor dimensions
  unified_linear::LinearOpType op_type =
      unified_linear::DetermineOpType(A, B, C);

  // Validate tensor dimensions
  unified_linear::ValidateTensorDims<T>(A, B, C, trans_A, trans_B);

  // Prepare output dimensions
  auto out_dims =
      unified_linear::PrepareOutputDims(A, B, C, trans_A, trans_B, op_type);
  out_D->Resize(out_dims);
  dev_ctx.template Alloc<T>(out_D);

  // Prepare output scale if needed
  if (out_D_scale) {
    out_D_scale->Resize(out_dims);
    dev_ctx.template Alloc<T>(out_D_scale);
  }

  // Determine optimal configuration
  unified_linear::OperationConfig config =
      unified_linear::DetermineOptimalConfig<T>(
          A, B, C, Bias, activation_type, dev_ctx);

#ifdef PADDLE_WITH_CUDA
  // Create scaled tensors
  unified_linear::ScaledTensor<T> A_scaled(
      &A, &A_scale, unified_linear::DetermineScaleType<T>(A, A_scale));
  unified_linear::ScaledTensor<T> B_scaled(
      &B, &B_scale, unified_linear::DetermineScaleType<T>(B, B_scale));
  unified_linear::ScaledTensor<T> C_scaled(
      &C, &C_scale, unified_linear::DetermineScaleType<T>(C, C_scale));

  // Dispatch to unified CUDA implementation
  switch (op_type) {
    case unified_linear::LinearOpType::kDot:
      phi::unified_linear::cuda::Dot(dev_ctx,
                                     A_scaled,
                                     B_scaled,
                                     static_cast<T>(alpha),
                                     static_cast<T>(beta),
                                     out_D);
      break;
    case unified_linear::LinearOpType::kMv:
      phi::unified_linear::cuda::Mv(dev_ctx,
                                    A_scaled,
                                    B_scaled,
                                    trans_A,
                                    static_cast<T>(alpha),
                                    static_cast<T>(beta),
                                    out_D,
                                    config);
      break;
    case unified_linear::LinearOpType::kMm:
      phi::unified_linear::cuda::Mm(dev_ctx,
                                    A_scaled,
                                    B_scaled,
                                    trans_A,
                                    trans_B,
                                    static_cast<T>(alpha),
                                    static_cast<T>(beta),
                                    out_D,
                                    config);
      break;
    case unified_linear::LinearOpType::kBmm:
      phi::unified_linear::cuda::Bmm(dev_ctx,
                                     A_scaled,
                                     B_scaled,
                                     trans_A,
                                     trans_B,
                                     static_cast<T>(alpha),
                                     static_cast<T>(beta),
                                     out_D,
                                     config);
      break;
    case unified_linear::LinearOpType::kLinear:
      phi::unified_linear::cuda::Linear(dev_ctx,
                                        A_scaled,
                                        B_scaled,
                                        Bias,
                                        trans_A,
                                        trans_B,
                                        static_cast<T>(alpha),
                                        static_cast<T>(beta),
                                        out_D,
                                        config,
                                        activation_type);
      break;
  }

  // Compute output scale if needed
  if (out_D_scale) {
    phi::unified_linear::cuda::ComputeOutputScale(
        dev_ctx, A_scaled, B_scaled, C_scaled, D_scale, out_D_scale);
  }
#else
  // CPU fallback implementation
  PADDLE_THROW(phi::errors::Unavailable(
      "UnifiedLinear is not supported on CPU. Please use CUDA."));
#endif
}

// Register kernels for different data types
PD_REGISTER_KERNEL(unified_linear,
                   GPU,
                   ALL_LAYOUT,
                   phi::UnifiedLinearKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

}  // namespace phi
