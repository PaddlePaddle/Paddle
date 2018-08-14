/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/attribute_type.h"

TEST(Attribute, TypeName) {
  bool boolean;
  int integer;
  float ft;
  std::string str;
  std::vector<bool> booleans;
  std::vector<int> integers;
  std::vector<std::string> strings;

  EXPECT_EQ("bool", paddle::framework::demangle(typeid(boolean).name()));
  EXPECT_EQ("int", paddle::framework::demangle(typeid(integer).name()));
  EXPECT_EQ("float", paddle::framework::demangle(typeid(ft).name()));
  EXPECT_EQ(
      "std::__cxx11::basic_string<char, std::char_traits<char>, "
      "std::allocator<char> >",
      paddle::framework::demangle(typeid(str).name()));
  EXPECT_EQ("std::vector<bool, std::allocator<bool> >",
            paddle::framework::demangle(typeid(booleans).name()));
  EXPECT_EQ("std::vector<int, std::allocator<int> >",
            paddle::framework::demangle(typeid(integers).name()));
  EXPECT_EQ(
      "std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, "
      "std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, "
      "std::char_traits<char>, std::allocator<char> > > >",
      paddle::framework::demangle(typeid(strings).name()));
}
