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

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/string_tensor.h"
#include "test/cpp/phi/core/allocator.h"

namespace phi {
namespace tests {

using pstring = ::phi::dtype::pstring;

TEST(string_tensor, ctor) {
  const DDim dims({1, 2});
  StringTensorMeta meta(dims);
  const auto string_allocator =
      std::make_unique<paddle::experimental::DefaultAllocator>(phi::CPUPlace());
  const auto alloc = string_allocator.get();
  auto check_string_tensor = [](const StringTensor& t,
                                const StringTensorMeta& m) -> bool {
    bool r{true};
    r = r && (t.numel() == product(m.dims));
    r = r && (t.dims() == m.dims);
    r = r && (t.place() == phi::CPUPlace());
    r = r && t.initialized();
    r = r && t.IsSharedWith(t);
    r = r && (t.meta() == m);
    return r;
  };
  auto cpu = CPUPlace();

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  CPUContext* cpu_ctx = reinterpret_cast<CPUContext*>(pool.Get(cpu));

  StringTensor tensor_0(alloc, meta);
  check_string_tensor(tensor_0, meta);

  pstring pshort_str = pstring("A short pstring.");
  pstring plong_str =
      pstring("A large pstring whose length is longer than 22.");

  pstring* data = cpu_ctx->template Alloc<pstring>(&tensor_0);
  data[0] = plong_str;
  data[1] = pshort_str;
  PADDLE_ENFORCE_EQ(tensor_0.data()[0],
                    plong_str,
                    common::errors::InvalidArgument(
                        "The tensor_0 should be equal to '%s', but got '%s'.",
                        plong_str,
                        tensor_0.data()[0]));
  PADDLE_ENFORCE_EQ(tensor_0.data()[1],
                    pshort_str,
                    common::errors::InvalidArgument(
                        "The tensor_0 should be equal to '%s', but got '%s'.",
                        pshort_str,
                        tensor_0.data()[1]));

  // Test Copy Constructor
  StringTensor tensor_1(tensor_0);
  PADDLE_ENFORCE_EQ(tensor_1.data()[0],
                    plong_str,
                    common::errors::InvalidArgument(
                        "The tensor_1 should be equal to '%s', but got '%s'.",
                        plong_str,
                        tensor_1.data()[0]));
  PADDLE_ENFORCE_EQ(tensor_1.data()[1],
                    pshort_str,
                    common::errors::InvalidArgument(
                        "The tensor_1 should be equal to '%s', but got '%s'.",
                        pshort_str,
                        tensor_1.data()[1]));

  // Test Copy Assignment
  StringTensor tensor_2(alloc, meta);
  tensor_2 = tensor_1;
  PADDLE_ENFORCE_EQ(tensor_2.data()[0],
                    plong_str,
                    common::errors::InvalidArgument(
                        "The tensor_2 should be equal to '%s', but got '%s'.",
                        plong_str,
                        tensor_2.data()[0]));
  PADDLE_ENFORCE_EQ(tensor_2.data()[1],
                    pshort_str,
                    common::errors::InvalidArgument(
                        "The tensor_2 should be equal to '%s', but got '%s'.",
                        pshort_str,
                        tensor_2.data()[1]));

  // Test Move Assignment
  StringTensor tensor_3(alloc, meta);
  tensor_3 = std::move(tensor_1);
  PADDLE_ENFORCE_EQ(tensor_3.data()[0],
                    plong_str,
                    common::errors::InvalidArgument(
                        "The tensor_3 should be equal to '%s', but got '%s'.",
                        plong_str,
                        tensor_3.data()[0]));
  PADDLE_ENFORCE_EQ(tensor_3.data()[1],
                    pshort_str,
                    common::errors::InvalidArgument(
                        "The tensor_3 should be equal to '%s', but got '%s'.",
                        pshort_str,
                        tensor_3.data()[1]));

  tensor_3.set_meta(meta);
}

TEST(pstring, func) {
  // Test Ctor
  pstring empty_str;
  pstring nchar_str(5, 'A');
  pstring copy_nchar_str(nchar_str);
  PADDLE_ENFORCE_EQ(
      empty_str,
      "",
      common::errors::InvalidArgument(
          "The empty_str should be empty, but got '%s'.", empty_str));
  PADDLE_ENFORCE_EQ(
      nchar_str,
      "AAAAA",
      common::errors::InvalidArgument(
          "The nchar_str should be 'AAAAA', but got '%s'.", nchar_str));
  PADDLE_ENFORCE_EQ(copy_nchar_str,
                    "AAAAA",
                    common::errors::InvalidArgument(
                        "The copy_nchar_str should be 'AAAAA', but got '%s'.",
                        copy_nchar_str));

  // Test Move Ctor
  pstring move_nchar_str(nchar_str);
  PADDLE_ENFORCE_EQ(move_nchar_str,
                    "AAAAA",
                    common::errors::InvalidArgument(
                        "The move_nchar_str should be 'AAAAA', but got '%s'.",
                        move_nchar_str));
  pstring std_str(std::string("BBBB"));
  PADDLE_ENFORCE_EQ(
      std_str,
      "BBBB",
      common::errors::InvalidArgument(
          "The std_str should be 'BBBB', but got '%s'.", std_str));

  pstring long_str = "A large pstring whose length is longer than 22.";
  pstring short_str = "A short pstring.";

  // Test operator+
  pstring plus_str = move_nchar_str + std_str;
  PADDLE_ENFORCE_EQ(
      plus_str,
      "AAAAABBBB",
      common::errors::InvalidArgument(
          "The plus_str should be 'AAAAABBBB', but got '%s'.", plus_str));

  // Test insert
  plus_str.insert(5, 1, 'C');
  PADDLE_ENFORCE_EQ(
      plus_str,
      "AAAAACBBBB",
      common::errors::InvalidArgument(
          "The plus_str should be 'AAAAABBBB', but got '%s'.", plus_str));
  plus_str.insert(5, "DDD", 0, 2);
  PADDLE_ENFORCE_EQ(
      plus_str,
      "AAAAADDCBBBB",
      common::errors::InvalidArgument(
          "The plus_str should be 'AAAAABBBB', but got '%s'.", plus_str));

  // Test pushback
  plus_str.push_back('E');
  PADDLE_ENFORCE_EQ(
      plus_str,
      "AAAAADDCBBBBE",
      common::errors::InvalidArgument(
          "The plus_str should be 'AAAAADDCBBBBE', but got '%s'.", plus_str));

  // Test append
  plus_str.append("FF");
  PADDLE_ENFORCE_EQ(
      plus_str,
      "AAAAADDCBBBBEFF",
      common::errors::InvalidArgument(
          "The plus_str should be 'AAAAADDCBBBBEFF', but got '%s'.", plus_str));
  plus_str.append(2, 'G');
  PADDLE_ENFORCE_EQ(
      plus_str,
      "AAAAADDCBBBBEFFGG",
      common::errors::InvalidArgument(
          "The plus_str should be 'AAAAADDCBBBBEFFGG', but got '%s'.",
          plus_str));

  // Test operator[]
  PADDLE_ENFORCE_EQ(
      long_str[0],
      'A',
      common::errors::InvalidArgument(
          "The long_str[0] should be 'A', but got '%s'.", long_str[0]));
  PADDLE_ENFORCE_EQ(
      short_str[0],
      'A',
      common::errors::InvalidArgument(
          "The short_str[0] should be 'A', but got '%s'.", short_str[0]));

  // Test capacity
  PADDLE_ENFORCE_EQ(short_str.capacity(),
                    22UL,
                    common::errors::InvalidArgument(
                        "The short_str's capacity should be 22, but got %d.",
                        short_str.capacity()));

  // Test reserve
  pstring reserve_str;
  PADDLE_ENFORCE_EQ(reserve_str.capacity(),
                    22UL,
                    common::errors::InvalidArgument(
                        "The reserve_str's capacity should be 22, but got %d.",
                        reserve_str.capacity()));
  // small -> large
  reserve_str.reserve(100);
  PADDLE_ENFORCE_EQ(reserve_str.capacity(),
                    111UL,
                    common::errors::InvalidArgument(
                        "The reserve_str's capacity should be 111, but got %d.",
                        reserve_str.capacity()));  // align(100) - 1 = 111
  // reserve more memory
  reserve_str.reserve(200);
  PADDLE_ENFORCE_EQ(reserve_str.capacity(),
                    207UL,
                    common::errors::InvalidArgument(
                        "The reserve_str's capacity should be 207, but got %d.",
                        reserve_str.capacity()));  // align(200) - 1 = 207

  // Test operator<<
  std::ostringstream oss1, oss2;
  oss1 << long_str;
  PADDLE_ENFORCE_EQ(
      oss1.str(),
      long_str,
      common::errors::InvalidArgument(
          "The oss1 should be '%s', but got '%s'.", long_str, oss1.str()));

  // Test iterator
  for (auto str_item : long_str) {
    oss2 << str_item;
  }
  PADDLE_ENFORCE_EQ(
      oss2.str(),
      long_str,
      common::errors::InvalidArgument(
          "The oss2 should be '%s', but got '%s'.", long_str, oss2.str()));

  // Test comparison operators
  PADDLE_ENFORCE_EQ((long_str < short_str),
                    true,
                    common::errors::InvalidArgument(
                        "The long_str should be less than short_str."));

  PADDLE_ENFORCE_EQ((long_str > short_str),
                    false,
                    common::errors::InvalidArgument(
                        "The long_str should not be greater than short_str."));
  PADDLE_ENFORCE_EQ((long_str == short_str),
                    false,
                    common::errors::InvalidArgument(
                        "The long_str should not be equal to short_str."));
  PADDLE_ENFORCE_EQ((long_str != short_str),
                    true,
                    common::errors::InvalidArgument(
                        "The long_str should not be equal to short_str."));
  PADDLE_ENFORCE_EQ((short_str < long_str),
                    false,
                    common::errors::InvalidArgument(
                        "The short_str should not be less than long_str."));
  PADDLE_ENFORCE_EQ((short_str > long_str),
                    true,
                    common::errors::InvalidArgument(
                        "The short_str should be greater than long_str."));
  PADDLE_ENFORCE_EQ((move_nchar_str < plus_str),
                    true,
                    common::errors::InvalidArgument(
                        "The move_nchar_str should be less than plus_str."));
  PADDLE_ENFORCE_EQ((plus_str > move_nchar_str),
                    true,
                    common::errors::InvalidArgument(
                        "The plus_str should be greater than move_nchar_str."));

  // Test empty
  PADDLE_ENFORCE_EQ(
      empty_str.empty(),
      true,
      common::errors::InvalidArgument("The empty_str should be empty."));
  PADDLE_ENFORCE_EQ(
      nchar_str.empty(),
      false,
      common::errors::InvalidArgument("The nchar_str should not be empty."));
  PADDLE_ENFORCE_EQ(empty_str.length(),
                    0UL,
                    common::errors::InvalidArgument(
                        "The empty_str's length should be 0, but got %d.",
                        empty_str.length()));

  // Test Resize
  nchar_str.resize(6, 'B');
  PADDLE_ENFORCE_EQ(
      nchar_str,
      "AAAAAB",
      common::errors::InvalidArgument(
          "The nchar_str should be 'AAAAAB', but got '%s'.", nchar_str));

  // Test operator =
  long_str = std::move(nchar_str);
  PADDLE_ENFORCE_EQ(
      long_str,
      "AAAAAB",
      common::errors::InvalidArgument(
          "The long_str should be 'AAAAAB', but got '%s'.", long_str));
  long_str = short_str;
  PADDLE_ENFORCE_EQ(
      short_str,
      long_str,
      common::errors::InvalidArgument(
          "The short_str should be '%s', but got '%s'.", long_str, short_str));
  short_str = 'A';
  PADDLE_ENFORCE_EQ(
      short_str,
      "A",
      common::errors::InvalidArgument(
          "The short_str should be 'A', but got '%s'.", short_str));
  short_str = std::move(copy_nchar_str);
  PADDLE_ENFORCE_EQ(
      short_str,
      "AAAAA",
      common::errors::InvalidArgument(
          "The short_str should be 'AAAAA', but got '%s'.", short_str));
}

}  // namespace tests
}  // namespace phi
