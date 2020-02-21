/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

static constexpr char predefined_cuda_functions_fp32[] = R"(
__device__ inline float real_exp(float x) { return ::expf(x); }
__device__ inline float real_log(float x) { return ::logf(x); }

)";

static constexpr char predefined_cuda_functions_fp64[] = R"(
__device__ inline double real_exp(double x) { return ::exp(x); }
__device__ inline double real_log(double x) { return ::log(x); }

)";

static constexpr char predefined_cuda_functions_fp16[] = R"(
__device__ inline float real_exp(float x) { return ::expf(x); }
__device__ inline float real_log(float x) { return ::logf(x); }

#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))

struct __align__(2) __half {
  __device__ __half() { }

 protected:
  unsigned short __x;
};

__device__ __half __float2half(const float f) {
  __half val;
  asm("{ cvt.rn.f16.f32 %0, %1; }\n" : "=h"(__HALF_TO_US(val)

) : "f"(f));
  return val;
}

__device__ float __half2float(const __half h) {
  float val;
  asm("{ cvt.f32.f16 %0, %1; }\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
  return val;
}

#undef __HALF_TO_US
#undef __HALF_TO_CUS

typedef __half float16;

)";

static constexpr char cuda_kernel_template_1d[] = R"(
extern "C" __global__ void $func_name($parameters) {
  for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
      idx < N;
      idx += gridDim.x * blockDim.x) {
    $compute_body
  }
}
)";

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle
