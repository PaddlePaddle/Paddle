// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/utils/variant.h"

#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/phi/core/enforce.h"

TEST(interface_test, type) {
  using phi::enforce::demangle;

  paddle::variant<bool, int, float, std::string, std::vector<int>> var;

  var = true;
  EXPECT_EQ(demangle(var.type().name()), "bool");

  var = 0;
  EXPECT_EQ(demangle(var.type().name()), "int");

  var = 0.f;
  EXPECT_EQ(demangle(var.type().name()), "float");

  var = std::string();
  EXPECT_EQ(demangle(var.type().name()),
            "std::__cxx11::basic_string<char, std::char_traits<char>, "
            "std::allocator<char> >");

  var = std::vector<int>();
  EXPECT_EQ(demangle(var.type().name()),
            "std::vector<int, std::allocator<int> >");
}
