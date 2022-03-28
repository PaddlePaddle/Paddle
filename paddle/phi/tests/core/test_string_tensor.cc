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

#include <sstream>
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

  tensor_3.set_meta(meta);
}

TEST(pstring, func) {
  // Test Ctor
  pstring empty_str;
  pstring nchar_str(5, 'A');
  pstring copy_nchar_str(nchar_str);
  CHECK_EQ(empty_str, "");
  CHECK_EQ(nchar_str, "AAAAA");
  CHECK_EQ(copy_nchar_str, "AAAAA");

  // Test Move Ctor
  pstring move_nchar_str(std::move(nchar_str));
  CHECK_EQ(move_nchar_str, "AAAAA");
  pstring std_str(std::string("BBBB"));
  CHECK_EQ(std_str, "BBBB");

  pstring long_str = "A large pstring whose length is longer than 22.";
  pstring short_str = "A short pstring.";

  // Test operator+
  pstring plus_str = move_nchar_str + std_str;
  CHECK_EQ(plus_str, "AAAAABBBB");

  // Test insert
  plus_str.insert(5, 1, 'C');
  CHECK_EQ(plus_str, "AAAAACBBBB");
  plus_str.insert(5, "DDD", 0, 2);
  CHECK_EQ(plus_str, "AAAAADDCBBBB");

  // Test pushback
  plus_str.push_back('E');
  CHECK_EQ(plus_str, "AAAAADDCBBBBE");

  // Test append
  plus_str.append("FF");
  CHECK_EQ(plus_str, "AAAAADDCBBBBEFF");
  plus_str.append(2, 'G');
  CHECK_EQ(plus_str, "AAAAADDCBBBBEFFGG");

  // Test operator[]
  CHECK_EQ(long_str[0], 'A');
  CHECK_EQ(short_str[0], 'A');

  // Test capacity
  CHECK_EQ(short_str.capacity(), 22UL);

  // Test reserve
  pstring reserve_str;
  CHECK_EQ(reserve_str.capacity(), 22UL);
  // small -> large
  reserve_str.reserve(100);
  CHECK_EQ(reserve_str.capacity(), 111UL);  // align(100) - 1 = 111
  // reserve more memory
  reserve_str.reserve(200);
  CHECK_EQ(reserve_str.capacity(), 207UL);  // align(200) - 1 = 207

  // Test operator<<
  std::ostringstream oss1, oss2;
  oss1 << long_str;
  CHECK_EQ(oss1.str(), long_str);

  // Test iterator
  for (auto it = long_str.begin(); it != long_str.end(); ++it) {
    oss2 << *it;
  }
  CHECK_EQ(oss2.str(), long_str);

  // Test comparision operators
  CHECK_EQ((long_str < short_str), true);
  CHECK_EQ((long_str > short_str), false);
  CHECK_EQ((long_str == short_str), false);
  CHECK_EQ((long_str != short_str), true);
  CHECK_EQ((short_str < long_str), false);
  CHECK_EQ((short_str > long_str), true);
  CHECK_EQ((move_nchar_str < plus_str), true);
  CHECK_EQ((plus_str > move_nchar_str), true);

  // Test empty
  CHECK_EQ(empty_str.empty(), true);
  CHECK_EQ(nchar_str.empty(), false);
  CHECK_EQ(empty_str.length(), 0UL);

  // Test Resize
  nchar_str.resize(6, 'B');
  CHECK_EQ(nchar_str, "AAAAAB");

  // Test operator =
  long_str = std::move(nchar_str);
  CHECK_EQ(long_str, "AAAAAB");
  long_str = short_str;
  CHECK_EQ(short_str, long_str);
  short_str = 'A';
  CHECK_EQ(short_str, "A");
  short_str = std::move(copy_nchar_str);
  CHECK_EQ(short_str, "AAAAA");
}

}  // namespace tests
}  // namespace phi
