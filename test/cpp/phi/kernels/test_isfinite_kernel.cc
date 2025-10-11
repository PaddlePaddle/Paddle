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

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/isfinite_kernel.h"

namespace phi {
namespace tests {
namespace {

const auto kCpuPlace = phi::CPUPlace();

const phi::CPUContext* GetCpuDeviceContext() {
  return phi::DeviceContextPool::Instance().GetByPlace(kCpuPlace);
}

template <typename T>
void FillTensor(const phi::DeviceContext* dev_ctx,
                phi::DenseTensor* tensor,
                const std::vector<T>& values) {
  tensor->Resize({static_cast<int64_t>(values.size())});
  auto* data = dev_ctx->template Alloc<T>(tensor);
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
}

template <typename T>
std::vector<bool> RunIsKernel(void (*kernel)(const phi::CPUContext&,
                                             const DenseTensor&,
                                             DenseTensor*),
                              const DenseTensor& input,
                              phi::DenseTensor* output,
                              const phi::CPUContext& ctx) {
  output->Resize(input.dims());
  kernel(ctx, input, output);
  const bool* result = output->data<bool>();
  return std::vector<bool>(result, result + input.numel());
}

}  // namespace

TEST(IsfiniteKernels, CpuIntegerTypes) {
  auto* dev_ctx = GetCpuDeviceContext();
  const auto& cpu_ctx = *dev_ctx;

  phi::DenseTensor input;
  FillTensor<int>(dev_ctx, &input, {0, 42, -7, 1024});

  phi::DenseTensor output;
  auto finite = RunIsKernel<int>(
      phi::IsfiniteKernel<int, phi::CPUContext>, input, &output, cpu_ctx);
  auto inf = RunIsKernel<int>(
      phi::IsinfKernel<int, phi::CPUContext>, input, &output, cpu_ctx);
  auto nan = RunIsKernel<int>(
      phi::IsnanKernel<int, phi::CPUContext>, input, &output, cpu_ctx);

  for (bool value : finite) {
    EXPECT_TRUE(value);
  }
  for (bool value : inf) {
    EXPECT_FALSE(value);
  }
  for (bool value : nan) {
    EXPECT_FALSE(value);
  }
}

TEST(IsfiniteKernels, CpuFloatTypes) {
  auto* dev_ctx = GetCpuDeviceContext();
  const auto& cpu_ctx = *dev_ctx;

  const float kInf = std::numeric_limits<float>::infinity();
  const float kNan = std::numeric_limits<float>::quiet_NaN();

  phi::DenseTensor float_input;
  FillTensor<float>(dev_ctx, &float_input, {0.0f, kInf, -kInf, kNan});

  phi::DenseTensor output;
  auto finite = RunIsKernel<float>(phi::IsfiniteKernel<float, phi::CPUContext>,
                                   float_input,
                                   &output,
                                   cpu_ctx);
  auto inf = RunIsKernel<float>(
      phi::IsinfKernel<float, phi::CPUContext>, float_input, &output, cpu_ctx);
  auto nan = RunIsKernel<float>(
      phi::IsnanKernel<float, phi::CPUContext>, float_input, &output, cpu_ctx);

  ASSERT_EQ(finite.size(), 4UL);
  EXPECT_TRUE(finite[0]);
  EXPECT_FALSE(finite[1]);
  EXPECT_FALSE(finite[2]);
  EXPECT_FALSE(finite[3]);

  ASSERT_EQ(inf.size(), 4UL);
  EXPECT_FALSE(inf[0]);
  EXPECT_TRUE(inf[1]);
  EXPECT_TRUE(inf[2]);
  EXPECT_FALSE(inf[3]);

  ASSERT_EQ(nan.size(), 4UL);
  EXPECT_FALSE(nan[0]);
  EXPECT_FALSE(nan[1]);
  EXPECT_FALSE(nan[2]);
  EXPECT_TRUE(nan[3]);

  phi::DenseTensor float16_input;
  std::vector<phi::float16> float16_values = {static_cast<phi::float16>(0.0f),
                                              static_cast<phi::float16>(kInf),
                                              static_cast<phi::float16>(-kInf),
                                              static_cast<phi::float16>(kNan)};
  FillTensor<phi::float16>(dev_ctx, &float16_input, float16_values);

  auto fp16_finite = RunIsKernel<phi::float16>(
      phi::IsfiniteKernel<phi::float16, phi::CPUContext>,
      float16_input,
      &output,
      cpu_ctx);
  auto fp16_inf =
      RunIsKernel<phi::float16>(phi::IsinfKernel<phi::float16, phi::CPUContext>,
                                float16_input,
                                &output,
                                cpu_ctx);
  auto fp16_nan =
      RunIsKernel<phi::float16>(phi::IsnanKernel<phi::float16, phi::CPUContext>,
                                float16_input,
                                &output,
                                cpu_ctx);

  ASSERT_EQ(fp16_finite.size(), 4UL);
  EXPECT_TRUE(fp16_finite[0]);
  EXPECT_FALSE(fp16_finite[1]);
  EXPECT_FALSE(fp16_finite[2]);
  EXPECT_FALSE(fp16_finite[3]);

  ASSERT_EQ(fp16_inf.size(), 4UL);
  EXPECT_FALSE(fp16_inf[0]);
  EXPECT_TRUE(fp16_inf[1]);
  EXPECT_TRUE(fp16_inf[2]);
  EXPECT_FALSE(fp16_inf[3]);

  ASSERT_EQ(fp16_nan.size(), 4UL);
  EXPECT_FALSE(fp16_nan[0]);
  EXPECT_FALSE(fp16_nan[1]);
  EXPECT_FALSE(fp16_nan[2]);
  EXPECT_TRUE(fp16_nan[3]);

  phi::DenseTensor bfloat16_input;
  std::vector<phi::bfloat16> bfloat16_values = {
      static_cast<phi::bfloat16>(0.0f),
      static_cast<phi::bfloat16>(kInf),
      static_cast<phi::bfloat16>(-kInf),
      static_cast<phi::bfloat16>(kNan)};
  FillTensor<phi::bfloat16>(dev_ctx, &bfloat16_input, bfloat16_values);

  auto bf16_finite = RunIsKernel<phi::bfloat16>(
      phi::IsfiniteKernel<phi::bfloat16, phi::CPUContext>,
      bfloat16_input,
      &output,
      cpu_ctx);
  auto bf16_inf = RunIsKernel<phi::bfloat16>(
      phi::IsinfKernel<phi::bfloat16, phi::CPUContext>,
      bfloat16_input,
      &output,
      cpu_ctx);
  auto bf16_nan = RunIsKernel<phi::bfloat16>(
      phi::IsnanKernel<phi::bfloat16, phi::CPUContext>,
      bfloat16_input,
      &output,
      cpu_ctx);

  ASSERT_EQ(bf16_finite.size(), 4UL);
  EXPECT_TRUE(bf16_finite[0]);
  EXPECT_FALSE(bf16_finite[1]);
  EXPECT_FALSE(bf16_finite[2]);
  EXPECT_FALSE(bf16_finite[3]);

  ASSERT_EQ(bf16_inf.size(), 4UL);
  EXPECT_FALSE(bf16_inf[0]);
  EXPECT_TRUE(bf16_inf[1]);
  EXPECT_TRUE(bf16_inf[2]);
  EXPECT_FALSE(bf16_inf[3]);

  ASSERT_EQ(bf16_nan.size(), 4UL);
  EXPECT_FALSE(bf16_nan[0]);
  EXPECT_FALSE(bf16_nan[1]);
  EXPECT_FALSE(bf16_nan[2]);
  EXPECT_TRUE(bf16_nan[3]);
}

TEST(IsfiniteKernels, CpuComplexTypes) {
  auto* dev_ctx = GetCpuDeviceContext();
  const auto& cpu_ctx = *dev_ctx;

  const float kInf = std::numeric_limits<float>::infinity();
  const float kNan = std::numeric_limits<float>::quiet_NaN();

  phi::DenseTensor complex_input;
  std::vector<phi::complex64> values = {phi::complex64(1.0f, 2.0f),
                                        phi::complex64(kInf, 0.0f),
                                        phi::complex64(0.0f, kNan),
                                        phi::complex64(kInf, kNan)};
  FillTensor<phi::complex64>(dev_ctx, &complex_input, values);

  phi::DenseTensor output;
  auto finite = RunIsKernel<phi::complex64>(
      phi::IsfiniteKernel<phi::complex64, phi::CPUContext>,
      complex_input,
      &output,
      cpu_ctx);
  auto inf = RunIsKernel<phi::complex64>(
      phi::IsinfKernel<phi::complex64, phi::CPUContext>,
      complex_input,
      &output,
      cpu_ctx);
  auto nan = RunIsKernel<phi::complex64>(
      phi::IsnanKernel<phi::complex64, phi::CPUContext>,
      complex_input,
      &output,
      cpu_ctx);

  ASSERT_EQ(finite.size(), 4UL);
  EXPECT_TRUE(finite[0]);
  EXPECT_FALSE(finite[1]);
  EXPECT_FALSE(finite[2]);
  EXPECT_FALSE(finite[3]);

  ASSERT_EQ(inf.size(), 4UL);
  EXPECT_FALSE(inf[0]);
  EXPECT_TRUE(inf[1]);
  EXPECT_FALSE(inf[2]);
  EXPECT_TRUE(inf[3]);

  ASSERT_EQ(nan.size(), 4UL);
  EXPECT_FALSE(nan[0]);
  EXPECT_FALSE(nan[1]);
  EXPECT_TRUE(nan[2]);
  EXPECT_TRUE(nan[3]);
}

}  // namespace tests
}  // namespace phi
