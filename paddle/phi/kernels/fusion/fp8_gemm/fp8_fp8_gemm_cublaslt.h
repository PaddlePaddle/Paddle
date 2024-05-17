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
#ifndef PADDLE_PHI_KERNELS_FUSION_FP8_GEMM_FP8_FP8_GEMM_CUBLASLT_H_
#define PADDLE_PHI_KERNELS_FUSION_FP8_GEMM_FP8_FP8_GEMM_CUBLASLT_H_

#include "paddle/phi/kernels/fusion/fp8_gemm/cublaslt_gemm.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename Context>
void cublaslt_fp8_fp8_fp16_gemm(
    const Context& ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& bias,
    bool transpose_x,
    bool transpose_y,
    const float scale,  // only support per-tensor quantization
    const std::string& activation_type,
    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      x.dims().size(), 2, "mat x for matmul fp8 just support 2-dim tensor");
  PADDLE_ENFORCE_EQ(
      y.dims().size(), 2, "mat y for matmul fp8 just support 2-dim tensor");
  PADDLE_ENFORCE_EQ(
      x.dims()[1], y.dims()[0], "x_dims[1] needs to equal to y_dims[0]");
  if (bias) {
    PADDLE_ENFORCE_EQ(bias->dims()[0],
                      y.dims()[1],
                      "bias_vecotr_dim needs to equal to y_dims[1]");
  }
  PADDLE_ENFORCE_EQ(x.dims()[1] % 16, 0, "fp8 matmul need x_dims[1] % 16 = 0.");
  PADDLE_ENFORCE_EQ(y.dims()[0] % 16, 0, "fp8 matmul need y_dims[0] % 16 = 0.");

  ctx.template Alloc<phi::dtype::float16>(out);
  CublasLtMatmulFP8<phi::dtype::float16>(
      ctx, x, y, scale, bias, activation_type, out);
}

template <typename Context>
void cublaslt_fp8_fp8_bf16_gemm(
    const Context& ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& bias,
    bool transpose_x,
    bool transpose_y,
    const float scale,  // only support per-tensor quantization
    const std::string& activation_type,
    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      x.dims().size(), 2, "mat x for matmul fp8 just support 2-dim tensor");
  PADDLE_ENFORCE_EQ(
      y.dims().size(), 2, "mat y for matmul fp8 just support 2-dim tensor");
  PADDLE_ENFORCE_EQ(
      x.dims()[1], y.dims()[0], "x_dims[1] needs to equal to y_dims[0]");
  if (bias) {
    PADDLE_ENFORCE_EQ(bias->dims()[0],
                      y.dims()[1],
                      "bias_vecotr_dim needs to equal to y_dims[1]");
  }
  PADDLE_ENFORCE_EQ(x.dims()[1] % 16, 0, "fp8 matmul need x_dims[1] % 16 = 0.");
  PADDLE_ENFORCE_EQ(y.dims()[0] % 16, 0, "fp8 matmul need y_dims[0] % 16 = 0.");

  ctx.template Alloc<phi::dtype::bfloat16>(out);
  CublasLtMatmulFP8<phi::dtype::bfloat16>(
      ctx, x, y, scale, bias, activation_type, out);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

#endif  // PADDLE_PHI_KERNELS_FUSION_FP8_GEMM_FP8_FP8_GEMM_CUBLASLT_H_
