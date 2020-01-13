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
#include <cuda_fp16.h>
typedef __half float16;

__device__ inline float16 real_exp(float16 x) { return ::hexp(x); }
__device__ inline float16 real_log(float16 x) { return ::hlog(x); }

)";

static constexpr char elementwise_cuda_template[] = R"(
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
