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

#include <string>
#include "gtest/gtest.h"

#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/tests/core/allocator.h"

namespace pten {
namespace tests {

using pstring = ::pten::dtype::pstring;

TEST(string_tensor, ctor) {
  const DDim dims({1, 2});
  StringTensorMeta meta(dims);
  const auto string_allocator =
      std::make_unique<paddle::experimental::StringAllocator>(
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
    return r;
  };

  StringTensor tensor_0(alloc, meta);
  check_string_tensor(tensor_0, meta);

  pstring pshort_str = pstring("A short pstring.");
  pstring plong_str =
      pstring("A large pstring whose length is longer than 22.");
  pstring* data = tensor_0.mutable_data();
  // need to init all pstrings, unless it will occur some unexpected segment
  // faults
  // memset(reinterpret_cast<char*>(data), 0, tensor_0.capacity());
  data[0] = plong_str;
  data[1] = pshort_str;
  CHECK_EQ(tensor_0.data()[0], plong_str);
  CHECK_EQ(tensor_0.data()[1], pshort_str);
}

}  // namespace tests
}  // namespace pten
