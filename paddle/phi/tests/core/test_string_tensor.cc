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

#include <string>
#include <utility>
#include "gtest/gtest.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/tests/core/allocator.h"

namespace phi {
namespace tests {

using pstring = ::phi::dtype::pstring;

TEST(string_tensor, ctor) {
  const DDim dims({1, 2});
  StringTensorMeta meta(dims);
  const auto string_allocator =
      std::make_unique<paddle::experimental::DefaultAllocator>(
          paddle::platform::CPUPlace());
  const auto alloc = string_allocator.get();
  auto check_string_tensor = [](const StringTensor& t,
                                const StringTensorMeta& m) -> bool {
    bool r{true};
    r = r && (t.numel() == product(m.dims));
    r = r && (t.dims() == m.dims);
    r = r && (t.place() == paddle::platform::CPUPlace());
    r = r && t.initialized();
    r = r && t.IsSharedWith(t);
    r = r && (t.meta() == m);
    return r;
  };
  auto cpu = CPUPlace();

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  CPUContext* cpu_ctx = reinterpret_cast<CPUContext*>(pool.Get(cpu));

  StringTensor tensor_0(alloc, meta);
  check_string_tensor(tensor_0, meta);

  pstring pshort_str = pstring("A short pstring.");
  pstring plong_str =
      pstring("A large pstring whose length is longer than 22.");

  pstring* data = cpu_ctx->template Alloc<pstring>(&tensor_0);
  data[0] = plong_str;
  data[1] = pshort_str;
  CHECK_EQ(tensor_0.data()[0], plong_str);
  CHECK_EQ(tensor_0.data()[1], pshort_str);

  // Test Copy Constructor
  StringTensor tensor_1(tensor_0);
  CHECK_EQ(tensor_1.data()[0], plong_str);
  CHECK_EQ(tensor_1.data()[1], pshort_str);

  // Test Copy Assignment
  StringTensor tensor_2(alloc, meta);
  tensor_2 = tensor_1;
  CHECK_EQ(tensor_2.data()[0], plong_str);
  CHECK_EQ(tensor_2.data()[1], pshort_str);

  // Test Move Assignment
  StringTensor tensor_3(alloc, meta);
  tensor_3 = std::move(tensor_1);
  CHECK_EQ(tensor_3.data()[0], plong_str);
  CHECK_EQ(tensor_3.data()[1], pshort_str);
}

TEST(pstring, dtype) {
  pstring long_str = "A large pstring whose length is longer than 22.";
  pstring short_str = "A short pstring.";
  // Test pstring assignment
  long_str = short_str;
  CHECK_EQ(long_str, short_str);
}

}  // namespace tests
}  // namespace phi
