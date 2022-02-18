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
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/kernels/strings/strings_copy_kernel.h"

namespace pten {
namespace tests {

using DDim = pten::framework::DDim;
using pstring = pten::dtype::pstring;

TEST(DEV_API, strings_copy) {
  // 1. create tensor
  const DDim dims({2, 3});
  StringTensorMeta meta(dims);

  pten::DeviceContextPool& pool = pten::DeviceContextPool::Instance();

  auto* dev_ctx =
      reinterpret_cast<pten::CPUContext*>(pool.Get(pten::CPUPlace()));

  const auto string_allocator =
      std::make_unique<paddle::experimental::DefaultAllocator>(
          pten::CPUPlace());
  const auto alloc = string_allocator.get();
  StringTensor string_src(alloc, meta);
  StringTensor string_dst(alloc, meta);

  // 2. Assign input text
  const char* input[] = {"A Short Pstring.",
                         "A Large Pstring Whose Length Is Longer Than 22.",
                         "abc",
                         "defg",
                         "hijklmn",
                         "opqrst"};
  pstring* string_src_data = string_src.mutable_data(dev_ctx->GetPlace());
  for (int i = 0; i < string_src.numel(); ++i) {
    string_src_data[i] = input[i];
  }
  pten::strings::Copy(*dev_ctx, string_src, false, &string_dst);
  for (int64_t i = 0; i < string_src.numel(); i++) {
    ASSERT_EQ(string_src.data()[i], string_dst.data()[i]);
  }
}

}  // namespace tests
}  // namespace pten
