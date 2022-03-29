/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <stdio.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#if (defined(__NVCC__) || defined(__HIPCC__))
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#endif

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/kernels/strings/strings_copy_kernel.h"
#include "paddle/phi/kernels/strings/strings_empty_kernel.h"
#include "paddle/phi/kernels/strings/strings_lower_upper_kernel.h"
namespace phi {
namespace tests {

namespace framework = paddle::framework;
using DDim = phi::DDim;
using pstring = ::phi::dtype::pstring;
using phi::GPUPlace;
using phi::CPUPlace;

TEST(DEV_API, strings_cast_convert) {
  auto gpu0 = GPUPlace();
  auto cpu = CPUPlace();

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  GPUContext* dev_ctx = reinterpret_cast<GPUContext*>(pool.Get(gpu0));
  CPUContext* cpu_ctx = reinterpret_cast<CPUContext*>(pool.Get(cpu));

  // 1. create tensor
  const DDim dims({1, 2});
  StringTensorMeta meta(dims);
  StringTensor gpu_strings_x = phi::strings::Empty(*dev_ctx, std::move(meta));
  StringTensor cpu_strings_x = phi::strings::Empty(*cpu_ctx, std::move(meta));
  StringTensor cpu_strings_lower_out =
      phi::strings::Empty(*cpu_ctx, std::move(meta));
  StringTensor cpu_strings_upper_out =
      phi::strings::Empty(*cpu_ctx, std::move(meta));

  std::string short_str = "A Short Pstring.";
  std::string long_str = "A Large Pstring Whose Length Is Longer Than 22.";

  pstring* cpu_strings_x_data =
      cpu_ctx->template Alloc<pstring>(&cpu_strings_x);
  cpu_strings_x_data[0] = short_str;
  cpu_strings_x_data[1] = long_str;

  phi::strings::Copy(*dev_ctx, cpu_strings_x, false, &gpu_strings_x);

  // 2. get expected results
  std::string expected_results[] = {short_str, short_str, long_str, long_str};
  std::transform(short_str.begin(),
                 short_str.end(),
                 expected_results[0].begin(),
                 ::tolower);
  std::transform(short_str.begin(),
                 short_str.end(),
                 expected_results[1].begin(),
                 ::toupper);
  std::transform(
      long_str.begin(), long_str.end(), expected_results[2].begin(), ::tolower);
  std::transform(
      long_str.begin(), long_str.end(), expected_results[3].begin(), ::toupper);

  // 3. test API, ascii encoding
  auto gpu_strings_lower_out =
      phi::strings::StringLower(*dev_ctx, gpu_strings_x, false);
  auto gpu_strings_upper_out =
      phi::strings::StringUpper(*dev_ctx, gpu_strings_x, false);

  phi::strings::Copy(
      *dev_ctx, gpu_strings_lower_out, false, &cpu_strings_lower_out);
  phi::strings::Copy(
      *dev_ctx, gpu_strings_upper_out, false, &cpu_strings_upper_out);

  // 4. check results
  ASSERT_EQ(gpu_strings_lower_out.numel(), 2);
  ASSERT_EQ(gpu_strings_upper_out.numel(), 2);
  const char* cpu_results[] = {cpu_strings_lower_out.data()[0].data(),
                               cpu_strings_upper_out.data()[0].data(),
                               cpu_strings_lower_out.data()[1].data(),
                               cpu_strings_upper_out.data()[1].data()};
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(cpu_results[i], expected_results[i]);
  }
}

TEST(DEV_API, strings_cast_convert_utf8) {
  auto gpu0 = GPUPlace();
  auto cpu = CPUPlace();

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  GPUContext* dev_ctx = reinterpret_cast<GPUContext*>(pool.Get(gpu0));
  CPUContext* cpu_ctx = reinterpret_cast<CPUContext*>(pool.Get(cpu));

  // 1. create tensor
  const DDim dims({1, 1});
  StringTensorMeta meta(dims);
  StringTensor gpu_strings_x = phi::strings::Empty(*dev_ctx, std::move(meta));
  StringTensor cpu_strings_x = phi::strings::Empty(*cpu_ctx, std::move(meta));
  StringTensor cpu_strings_lower_out =
      phi::strings::Empty(*cpu_ctx, std::move(meta));
  StringTensor cpu_strings_upper_out =
      phi::strings::Empty(*cpu_ctx, std::move(meta));
  std::string utf8_str = "óÓsscHloëË";
  pstring* cpu_strings_x_data =
      cpu_ctx->template Alloc<pstring>(&cpu_strings_x);

  cpu_strings_x_data[0] = utf8_str;
  phi::strings::Copy(*dev_ctx, cpu_strings_x, false, &gpu_strings_x);

  // 2. get expected results
  std::string expected_results[] = {"óósschloëë", "ÓÓSSCHLOËË"};

  // 3. test API, ascii encoding
  auto gpu_strings_lower_out =
      phi::strings::StringLower(*dev_ctx, gpu_strings_x, true);
  auto gpu_strings_upper_out =
      phi::strings::StringUpper(*dev_ctx, gpu_strings_x, true);
  phi::strings::Copy(
      *dev_ctx, gpu_strings_lower_out, false, &cpu_strings_lower_out);
  phi::strings::Copy(
      *dev_ctx, gpu_strings_upper_out, false, &cpu_strings_upper_out);

  // 4. check results
  const char* cpu_results[] = {cpu_strings_lower_out.data()[0].data(),
                               cpu_strings_upper_out.data()[0].data()};
  ASSERT_EQ(cpu_strings_lower_out.numel(), 1);
  ASSERT_EQ(cpu_strings_upper_out.numel(), 1);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(cpu_results[i], expected_results[i]);
  }
}

}  // namespace tests
}  // namespace phi
