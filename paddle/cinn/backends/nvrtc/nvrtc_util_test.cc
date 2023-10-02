// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/nvrtc/nvrtc_util.h"

#include <gtest/gtest.h>

namespace cinn {
namespace backends {
namespace nvrtc {

TEST(Compiler, basic) {
  Compiler compiler;

  std::string source_code = R"ROC(
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = a * x[tid] + y[tid];
  }
}
)ROC";

  auto ptx = compiler(source_code);

  LOG(INFO) << "ptx:\n" << ptx;
}

TEST(Compiler, float16) {
  Compiler compiler;

  std::string source_code = R"(
#include <cstdint>
#define CINN_WITH_CUDA
#include "float16.h"
using cinn::common::float16;

extern "C" __global__
void cast_fp32_to_fp16_cuda_kernel(const float* input, const int num, float16* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    out[idx] = float16(input[idx]);
  }
}
)";

  auto ptx = compiler(source_code);

  LOG(INFO) << "ptx:\n" << ptx;
}

TEST(Compiler, bfloat16) {
  Compiler compiler;

  std::string source_code = R"(
#include <cstdint>
#define CINN_WITH_CUDA
#include "bfloat16.h"
using cinn::common::bfloat16;

extern "C" __global__
void cast_fp32_to_bf16_cuda_kernel(const float* input, const int num, bfloat16* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num) {
    out[idx] = bfloat16(input[idx]);
  }
}
)";

  auto ptx = compiler(source_code);

  LOG(INFO) << "ptx:\n" << ptx;
}

}  // namespace nvrtc
}  // namespace backends
}  // namespace cinn
