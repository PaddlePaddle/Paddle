// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #include "paddle/extension.h"
#include <cuda_fp8.h>
#include <cstdint>
#include <vector>
#include "paddle/common/flags.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/empty_kernel.h"  // NOLINT

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"

COMMON_DECLARE_bool(enable_pir_api);

namespace phi {

template <typename Context,
          bool using_1x128_vec_quant,
          bool input_transpose,
          bool output_scale_transpose,
          bool using_pow2_scale>
void FP8QuantBlockWiseKernelImpl(const Context &dev_ctx,
                                 const DenseTensor &X,
                                 DenseTensor *out,
                                 DenseTensor *scale,
                                 DenseTensor *out_transposed,
                                 DenseTensor *scale_transposed) {
  // using namespace cute;
  const size_t src_rows = X.dims()[0];
  const size_t src_cols = X.dims()[1];
  const size_t quanted_cols = scale.dims()[1];
  const size_t quanted_rows = scale_transposed.dims()[1];

  dim3 block(32, 32);
  // Assuming src_rows and src_cols are multiples of 128
  dim3 grid(src_rows / 32, src_cols / 32);

  auto kernel = using_1x128_vec_quant
                    ? quantize_1x128_kernel<input_transpose,
                                            output_scale_transpose,
                                            using_pow2_scale>
                    : quantize_128x128_kernel<input_transpose,
                                              output_scale_transpose,
                                              using_pow2_scale>;
  kernel<<<grid, block, 0, dev_ctx.stream()>>>();
}

// T is x's input type and out_dtype is in args
template <typename T, typename Context>
void FP8QuantBlockWiseKernel(const Context &dev_ctx,
                             const DenseTensor &X,
                             bool using_1x128_vec_quant,
                             bool input_transpose,
                             bool output_scale_transpose,
                             bool using_e5m2,
                             bool using_pow2_scale,
                             DenseTensor *out,
                             DenseTensor *scale,
                             DenseTensor *out_transposed,
                             DenseTensor *scale_transposed) {
  PD_CHECK(X.dtype() == phi::DataType::BFLOAT16,
           "X datatype error, can only be bfloat16");

  dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out);
  dev_ctx.template Alloc<float>(scale);
  if (input_transpose) {
    dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out_transposed);
    dev_ctx.template Alloc<float>(scale_transposed);
  }
#define DISPATCH_BOOL(condition, ConstName, ...) \
  {                                              \
    if (condition) {                             \
      constexpr bool ConstName = true;           \
      { __VA_ARGS__ }                            \
    } else {                                     \
      constexpr bool ConstName = false;          \
      { __VA_ARGS__ }                            \
    }                                            \
  }
  // Currently we only support bfloat16 as input type,
  // fp8_e4m3fn as output type.
  DISPATCH_BOOL(
      using_1x128_vec_quant,
      k_using_1x128_vec_quant,
      DISPATCH_BOOL(
          input_transpose,
          k_input_transpose,
          DISPATCH_BOOL(
              output_scale_transpose,
              k_output_scale_transpose,
              DISPATCH_BOOL(
                  using_pow2_scale,
                  k_using_pow2_scale,
                  FP8QuantBlockWiseKernelImpl<Context,
                                              k_using_1x128_vec_quant,
                                              k_input_transpose,
                                              k_output_scale_transpose,
                                              k_using_pow2_scale>(
                      dev_ctx,
                      X,
                      out,
                      scale,
                      out_transposed,
                      scale_transposed);))));
#undef DISPATCH_BOOL
}
}  // namespace phi

PD_REGISTER_KERNEL(fp8_quant_blockwise,
                   GPU,
                   ALL_LAYOUT,
                   phi::FP8QuantBlockWiseKernel,
                   phi::bfloat16,
                   float,
                   double) {}
