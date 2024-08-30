// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#if CUDA_VERSION >= 12010
#include "paddle/phi/kernels/fusion/fp8_gemm/fp8_gemm_with_cublasLt/cublaslt_gemm.h"
#endif
namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename InputType, typename Context>
void fp8_fp8_half_gemm(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& bias,
    const bool trans_x,
    const bool trans_y,
    const float scale,  // only support per-tensor quantization
    const std::string& output_dtype,
    const std::string& activation_type,
    DenseTensor* out) {
#if CUDA_VERSION >= 12010
  VLOG(3) << "fp8_fp8_half_gemm_fused of cublasLt start run: ";
  static_assert(std::is_same<Context, phi::GPUContext>::value,
                "fp8_fp8_gemm must be in GPU");
  if (out->dtype() == phi::DataType::BFLOAT16) {
    cublaslt_fp8_fp8_bf16_gemm<Context>(
        dev_ctx, x, y, bias, trans_x, trans_y, scale, activation_type, out);
  } else if (out->dtype() == phi::DataType::FLOAT16) {
    cublaslt_fp8_fp8_fp16_gemm<Context>(
        dev_ctx, x, y, bias, trans_x, trans_y, scale, activation_type, out);
  } else {
    PADDLE_THROW(common::errors::Fatal(
        "fp8_fp8_half_gemm_fused only support bfloat16 and float16 output"));
  }

#else
  PADDLE_THROW(common::errors::Fatal(
      "fp8_fp8_half_gemm_fused need CUDA 12.1+ and sm_89+"));
#endif
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fp8_fp8_half_gemm_fused,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::fp8_fp8_half_gemm,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2) {}
