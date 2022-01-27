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
#include <algorithm>
#include <memory>
#include <string>

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/kernels/strings/case_convert_kernel.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;
using pstring = ::pten::dtype::pstring;

TEST(DEV_API, strings_cast_convert) {
  // 1. create tensor
  const DDim dims({1, 2});
  StringTensorMeta meta(dims);
  const auto string_allocator =
      std::make_unique<paddle::experimental::StringAllocator>(
          paddle::platform::CPUPlace());
  const auto alloc = string_allocator.get();
  StringTensor dense_x(alloc, meta);

  std::string short_str = "A Short Pstring.";
  std::string long_str = "A Large Pstring Whose Length Is Longer Than 22.";

  pstring* dense_x_data = dense_x.mutable_data();
  dense_x_data[0] = short_str;
  dense_x_data[1] = long_str;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

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
  auto dense_lower_out = pten::strings::StringLower(
      *(static_cast<pten::CPUContext*>(dev_ctx)), "", dense_x);
  auto dense_upper_out = pten::strings::StringUpper(
      *(static_cast<pten::CPUContext*>(dev_ctx)), "", dense_x);

  // 4. check results
  ASSERT_EQ(dense_lower_out.numel(), 2);
  ASSERT_EQ(dense_upper_out.numel(), 2);

  // lower case
  ASSERT_EQ(dense_lower_out.data()[0].data(), expected_results[0]);
  ASSERT_EQ(dense_lower_out.data()[1].data(), expected_results[2]);

  // upper case
  ASSERT_EQ(dense_upper_out.data()[0].data(), expected_results[1]);
  ASSERT_EQ(dense_upper_out.data()[1].data(), expected_results[3]);
}

TEST(DEV_API, strings_cast_convert_utf8) {
  // 1. create tensor
  const DDim dims({1, 1});
  StringTensorMeta meta(dims);

  const auto string_allocator =
      std::make_unique<paddle::experimental::StringAllocator>(
          paddle::platform::CPUPlace());
  const auto alloc = string_allocator.get();
  StringTensor dense_x(alloc, meta);

  std::string utf8_str = "óÓsscHloëË";

  pstring* dense_x_data = dense_x.mutable_data();
  dense_x_data[0] = utf8_str;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. get expected results
  std::string expected_results[] = {"óósschloëë", "ÓÓSSCHLOËË"};

  // 3. test API, ascii encoding
  auto dense_lower_out = pten::strings::StringLower(
      *(static_cast<pten::CPUContext*>(dev_ctx)), "utf-8", dense_x);
  auto dense_upper_out = pten::strings::StringUpper(
      *(static_cast<pten::CPUContext*>(dev_ctx)), "utf-8", dense_x);

  // 4. check results
  ASSERT_EQ(dense_lower_out.numel(), 1);
  ASSERT_EQ(dense_upper_out.numel(), 1);

  // lower case
  ASSERT_EQ(dense_lower_out.data()[0].data(), expected_results[0]);

  // upper case
  ASSERT_EQ(dense_upper_out.data()[0].data(), expected_results[1]);
}

}  // namespace tests
}  // namespace pten
