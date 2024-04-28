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
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/backends/dynload/fp8_gemm_fused.h"
#include "paddle/phi/backends/gpu/gpu_context.h"


namespace phi{
namespace fusion{
namespace cutlass_internal{

template <typename OutputType, typename Context>
void FP8FP8Gemm(const Context& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              const paddle::optional<DenseTensor>& bias,
              const float scale, // only support per-tensor quantization
              const int x_num_col_dims,
              const int y_num_col_dims,
              const std::string& activation_type,
              const bool padding_weights,
              DenseTensor* out){

  dev_ctx.template Alloc<OutputType>(out);
  const DenseTensor x_matrix =
      x.dims().size() > 2 ? ReshapeToMatrix(x, x_num_col_dims) : x;
  const DenseTensor y_matrix =
      y.dims().size() > 2 ? ReshapeToMatrix(y, y_num_col_dims) : y;

  const int M = x_matrix.dims()[0];
  const int N = y_matrix.dims()[1];
  const int K = x_matrix.dims()[1];

  const int lda = x_matrix.dims()[1];
  const int ldb = y_matrix.dims()[1];
  const int ldd = out->dims()[1];

  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  void* dlhandler = phi::dynload::GetFP8FP8GemmFusedHandle();
  func fp8_gemm_func = NULL;
  if(!dlhandler){
    PADDLE_THROW(phi::errors::InvalidArgument(
          "The library of fp8 phi kernels has not been compiled. Please run the compile.sh script according to README to compile the cutlass kernels library"));
  }
  auto *bias_ptr = bias.get_ptr();
  if(bias_ptr){
    auto *bias_data = bias->data<float>();
    auto bias_dims = common::vectorize(bias->dims());

    GemmEpilogueAllParams params = {
        reinterpret_cast<const void*>(x_matrix.data<phi::dtype::float8_e4m3fn>()),
        reinterpret_cast<const void*>(y_matrix.data<phi::dtype::float8_e4m3fn>()),
        reinterpret_cast<void*>(out->data<OutputType>()),
        scale,
        M,
        N,
        K,
        lda,
        ldb,
        ldd,
        dev_ctx.stream(),
        device_id,
        sm_version,
        0.01,       // for leaky_relu
        reinterpret_cast<const void*>(bias_data),
        &bias_dims
    };
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
  }else{
    GemmEpilogueAllParams params = {
        reinterpret_cast<const void*>(x_matrix.data<phi::dtype::float8_e4m3fn>()),
        reinterpret_cast<const void*>(y_matrix.data<phi::dtype::float8_e4m3fn>()),
        reinterpret_cast<void*>(out->data<OutputType>()),
        scale,
        M,
        N,
        K,
        lda,
        ldb,
        ldd,
        dev_ctx.stream(),
        device_id,
        sm_version,
        0.01,       // for leaky_relu
        nullptr,
        nullptr
    };
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

PD_REGISTER_KERNEL(fp8_fp8_gemm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::FP8FP8Gemm,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
