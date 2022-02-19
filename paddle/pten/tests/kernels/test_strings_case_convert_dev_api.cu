/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/kernels/strings/case_convert_kernel.h"
#include "paddle/pten/kernels/strings/strings_copy_kernel.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = pten::framework::DDim;
using pstring = ::pten::dtype::pstring;
using paddle::platform::CPUPlace;
using paddle::platform::CUDAPlace;
using paddle::platform::CUDADeviceContext;

__global__ void CopyFromVec(pstring* dst, char** src, const int64_t num) {
  CUDA_KERNEL_LOOP(i, num) { dst[i] = pstring(src[i]); }
}

__global__ void CopyToVec(char** dst, pstring** src, const int64_t num) {
  CUDA_KERNEL_LOOP(i, num) {
    // Copy from gpu to cpu
    memcpy(dst[i], src[i]->data(), src[i]->size() + 1);
  }
}

TEST(DEV_API, strings_cast_convert) {
  auto gpu0 = CUDAPlace();
  auto cpu = CPUPlace();

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  GPUContext* dev_ctx = reinterpret_cast<GPUContext*>(pool.Get(gpu0));
  CPUContext* cpu_ctx = reinterpret_cast<CPUContext*>(pool.Get(cpu));

  // 1. create tensor
  const DDim dims({1, 2});
  StringTensorMeta meta(dims);
  StringTensor gpu_strings_x = pten::strings::Empty(*dev_ctx, std::move(meta));
  StringTensor cpu_strings_x = pten::strings::Empty(*cpu_ctx, std::move(meta));
  StringTensor cpu_strings_lower_out =
      pten::strings::Empty(*cpu_ctx, std::move(meta));
  StringTensor cpu_strings_upper_out =
      pten::strings::Empty(*cpu_ctx, std::move(meta));

  std::string short_str = "A Short Pstring.";
  std::string long_str = "A Large Pstring Whose Length Is Longer Than 22.";
  pstring* cpu_strings_x_data = cpu_strings_x.mutable_data(cpu);
  cpu_strings_x_data[0] = short_str;
  cpu_strings_x_data[1] = long_str;

  pten::strings::Copy(*dev_ctx, cpu_strings_x, false, &gpu_strings_x);

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
      pten::strings::StringLower(*dev_ctx, "", gpu_strings_x);
  auto gpu_strings_upper_out =
      pten::strings::StringUpper(*dev_ctx, "", gpu_strings_x);

  pten::strings::Copy(
      *dev_ctx, gpu_strings_lower_out, false, &cpu_strings_lower_out);
  pten::strings::Copy(
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

// TEST(DEV_API, strings_cast_convert_utf8) {
//   auto gpu0 = CUDAPlace();
//   // 1. create tensor
//   const DDim dims({1, 1});
//   StringTensorMeta meta(dims);

//   const auto string_allocator =
//       std::make_unique<paddle::experimental::DefaultAllocator>(gpu0);
//   const auto alloc = string_allocator.get();
//   StringTensor dense_x(alloc, meta);

//   std::string utf8_str = "óÓsscHloëË";

//   pstring* dense_x_data = dense_x.mutable_data();
//   dense_x_data[0] = utf8_str;

//   paddle::platform::DeviceContextPool& pool =
//       paddle::platform::DeviceContextPool::Instance();
//   pten::GPUContext* dev_ctx =
//   reinterpret_cast<pten::GPUContext*>(pool.Get(gpu0));

//   // 2. get expected results
//   std::string expected_results[] = {"óósschloëë", "ÓÓSSCHLOËË"};
//   std::string cpu_results[] = {"", ""};
//   for (int i = 0; i < 2; ++i) {
//       cpu_results[i] = std::string(expected_results[i].length(), 0);
//   }
//   // 3. test API, ascii encoding
//   auto dense_lower_out = pten::StringLower(
//       *dev_ctx,
//       "utf-8",
//       dense_x);
//   auto dense_upper_out = pten::StringUpper(
//       *dev_ctx,
//       "utf-8",
//       dense_x);

//   // 4. check results
//   ASSERT_EQ(dense_lower_out.numel(), 1);
//   ASSERT_EQ(dense_upper_out.numel(), 1);

//   paddle::memory::Copy(CPUPlace(), const_cast<char*>(cpu_results[0].data()),
//   gpu0, dense_lower_out.data()[0].data(), cpu_results[0].length(),
//   dev_ctx->stream());
//   paddle::memory::Copy(CPUPlace(), const_cast<char*>(cpu_results[1].data()),
//   gpu0, dense_upper_out.data()[0].data(), cpu_results[1].length(),
//   dev_ctx->stream());

//   for (int i = 0; i < 2; ++i) {
//       ASSERT_EQ(cpu_results[i], expected_results[i]);
//   }
// }

}  // namespace tests
}  // namespace pten
