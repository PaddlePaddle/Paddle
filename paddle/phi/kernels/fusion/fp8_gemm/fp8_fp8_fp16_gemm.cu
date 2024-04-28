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
#include "./cutlass_kernels/fp8_fp8_gemm_scale_bias_act.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/backends/dynload/fp8_gemm_fused.h"

namespace phi{
namespace fusion{
namespace cutlass_internal{

template <typename InputType, typename Context>
void fp8_fp8_fp16_gemm(const Context& dev_ctx,
              const DenseTensor& x,
              const DenseTensor& y,
              const paddle::optional<DenseTensor>& bias,
              const bool trans_x,
              const bool trans_y,
              const float scale, // only support per-tensor quantization
              const std::string& activation_type,
              DenseTensor* out){
  static_assert(std::is_same<Context, phi::GPUContext>::value,
    "fp8_fp8_gemm must be in GPU");

  dev_ctx.template Alloc<phi::dtype::float16>(out);
  auto place = dev_ctx.GetPlace();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(dev_ctx.stream());
  int64_t device_id = place.GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  int M=0;
  int K=0;
  int N=0;
  int lda = x.dims()[1];
  int ldb = y.dims()[1];
  int ldd = out->dims()[1];
  if(!trans_x){
    M = x.dims()[0];
    K = x.dims()[1];

  }else{
    M = x.dims()[1];
    K = x.dims()[0];
  }

  if(!trans_y){
    N = y.dims()[1];
  }else{
    N = y.dims()[0];
  }

  void* dlhandler = phi::dynload::GetFP8FP8GemmFusedHandle();
  if(!dlhandler){
    PADDLE_THROW(phi::errors::InvalidArgument(
          "The library of fp8 phi kernels has not been compiled. Please run the compile.sh script according to README to compile the cutlass kernels library"));
  }


  std::string input_dtype;
  if(x.dtype() == phi::DataType::FLOAT8_E4M3FN){
    input_dtype = "e4m3";
  }else{
    input_dtype = "e5m2";
  }
  std::string output_dtype = "fp16";

  void *bias_data = nullptr;
  std::vector<int64_t> bias_dims{};
  if(bias){
    bias_data = reinterpret_cast<void*>(const_cast<float*>(bias.get().data<float>()));
    bias_dims = common::vectorize(bias.get().dims());
  }
  GemmEpilogueAllParams params = {
      reinterpret_cast<const void*>(x.data<InputType>()),
      reinterpret_cast<const void*>(y.data<InputType>()),
      reinterpret_cast<void*>(out->data<phi::dtype::float16>()),
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
      bias_data,
      &bias_dims,
      input_dtype,
      output_dtype,
      activation_type,
  };
    func fp8_gemm_func = (func)(dlsym(dlhandler, "fp8_fp8_gemm_scale_bias_act"));
  fp8_gemm_func(params);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fp8_fp8_fp16_gemm_fused,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::fp8_fp8_fp16_gemm,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2) {}
