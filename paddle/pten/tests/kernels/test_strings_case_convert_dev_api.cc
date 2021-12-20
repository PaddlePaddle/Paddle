
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

#include "paddle/pten/include/strings.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/platform/pstring.h"

namespace pten {
namespace tests {

namespace framework = paddle::framework;
using DDim = paddle::framework::DDim;
using pstring = ::pten::platform::pstring;

TEST(DEV_API, strings_cast_convert) {
  // 1. create tensor
  const DDim dims({1, 2});
  const DataType dtype{DataType::STRING};
  const DataLayout layout{DataLayout::NHWC};
  const std::vector<std::vector<size_t>> lod{};
  DenseTensorMeta meta(dtype, dims, layout, lod);

  const auto alloc = std::make_shared<paddle::experimental::StringAllocator>(
      paddle::platform::CPUPlace());
  DenseTensor dense_x(alloc, meta);

  std::string short_str = "A Short Pstring.";
  std::string long_str = "A Large Pstring Whose Length Is Longer Than 22.";

  pstring* dense_x_data = dense_x.mutable_data<pstring>();
  dense_x_data[0] = short_str;
  dense_x_data[1] = long_str;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(paddle::platform::CPUPlace());

  // 2. test API
  auto dense_lower_out = pten::StringLower(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
      "",
      dense_x);
  auto dense_upper_out = pten::StringUpper(
      *(static_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)),
      "",
      dense_x);

  // 3. check result
  ASSERT_EQ(dense_lower_out.dims().size(), 2);
  ASSERT_EQ(dense_upper_out.dims().size(), 2);

  // lower case
  std::transform(
      short_str.begin(), short_str.end(), short_str.begin(), ::tolower);
  std::transform(long_str.begin(), long_str.end(), long_str.begin(), ::tolower);
  ASSERT_EQ(dense_lower_out.data<pstring>()[0].data(), short_str);
  ASSERT_EQ(dense_lower_out.data<pstring>()[1].data(), long_str);

  // upper case
  std::transform(
      short_str.begin(), short_str.end(), short_str.begin(), ::toupper);
  std::transform(long_str.begin(), long_str.end(), long_str.begin(), ::toupper);
  ASSERT_EQ(dense_upper_out.data<pstring>()[0].data(), short_str);
  ASSERT_EQ(dense_upper_out.data<pstring>()[1].data(), long_str);
}

}  // namespace tests
}  // namespace pten
