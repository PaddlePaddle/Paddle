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

#include "fp8_common.h"
#include "paddle/phi/backends/dynload/fp8_gemm_fused.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename InputType, typename Context>
void FP8FP8BF16Gemm(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const paddle::optional<DenseTensor>& bias,
    const bool trans_x,
    const bool trans_y,
    const float scale,  // only support per-tensor quantization
    const std::string& activation_type,
    DenseTensor* out) {
  static_assert(std::is_same<Context, phi::GPUContext>::value,
                "fp8_fp8_gemm must be in GPU");

  dev_ctx.template Alloc<phi::dtype::bfloat16>(out);
  auto place = dev_ctx.GetPlace();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(dev_ctx.stream());
  int64_t device_id = place.GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  const int M = x.dims()[0];
  const int N = y.dims()[1];
  const int K = x.dims()[1];

  const int lda = x.dims()[1];
  const int ldb = y.dims()[1];
  const int ldd = out->dims()[1];

  void* dlhandler = phi::dynload::GetFP8FP8GemmFusedHandle();
  func fp8_gemm_func = NULL;
  if (!dlhandler) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The library of fp8 phi kernels has not been compiled. Please run the "
        "compile.sh script according to README to compile the cutlass kernels "
        "library"));
  }

  std::string input_dtype;
  if (x.dtype() == phi::DataType::FLOAT8_E4M3FN) {
    input_dtype = "e4m3";
  } else {
    input_dtype = "e5m2";
  }
  std::string output_dtype = "bf16";

  if(bias){
    auto* bias_data = bias.get().data<float>();
    auto bias_dims = common::vectorize(bias.get().dims());

    GemmEpilogueAllParams params = {
        reinterpret_cast<const void*>(x.data<InputType>()),
        reinterpret_cast<const void*>(y.data<InputType>()),
        reinterpret_cast<void*>(out->data<phi::dtype::bfloat16>()),
        scale,
        M,
        N,
        K,
        lda,
        ldb,
        ldd,
        place,
        stream,
        sm_version,
        0.01,  // for leaky_relu
        reinterpret_cast<const void*>(bias_data),
        &bias_dims,
        input_dtype,
        output_dtype};
    if (activation_type == "identity" || activation_type == "") {
      fp8_gemm_func = (func)(dlsym(dlhandler, "FP8FP8GemmScaleBias"));
    } else if (activation_type == "relu") {
      fp8_gemm_func = (func)(dlsym(dlhandler, "FP8FP8GemmScaleBiasRelu"));
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "fp8_fp8_gemm does not support this activation_type: %s.",
          activation_type.c_str()));
    }
    fp8_gemm_func(params);
  } else {
    GemmEpilogueAllParams params = {
        reinterpret_cast<const void*>(x.data<InputType>()),
        reinterpret_cast<const void*>(y.data<InputType>()),
        reinterpret_cast<void*>(out->data<phi::dtype::bfloat16>()),
        scale,
        M,
        N,
        K,
        lda,
        ldb,
        ldd,
        place,
        stream,
        sm_version,
        0.01,  // for leaky_relu
        nullptr,
        nullptr,
        input_dtype,
        output_dtype};
    if (activation_type == "identity" || activation_type == "") {
      fp8_gemm_func = (func)(dlsym(dlhandler, "FP8FP8GemmScale"));
    } else if (activation_type == "relu") {
      fp8_gemm_func = (func)(dlsym(dlhandler, "FP8FP8GemmScaleRelu"));
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "fp8_fp8_gemm does not support this activation_type: %s.",
          activation_type.c_str()));
    }
    fp8_gemm_func(params);
  }
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fp8_fp8_bf16_gemm_fused,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::FP8FP8BF16Gemm,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2) {}
