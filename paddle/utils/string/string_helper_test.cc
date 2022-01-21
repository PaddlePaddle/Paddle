// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/utils/string/string_helper.h"

#include <string>

#include "gtest/gtest.h"

TEST(StringHelper, EndsWith) {
  std::string input("hello world");
  std::string test1("world");
  std::string test2("helloworld");
  std::string test3("hello world hello world");

  EXPECT_TRUE(paddle::string::ends_with(input, test1));
  EXPECT_TRUE(paddle::string::ends_with(input, input));

  EXPECT_FALSE(paddle::string::ends_with(input, test2));
  EXPECT_FALSE(paddle::string::ends_with(input, test3));
}

TEST(StringHelper, FormatStringAppend) {
  std::string str("hello");
  char fmt[] = "%d";

  paddle::string::format_string_append(str, fmt, 10);
  EXPECT_EQ(str, "hello10");
}

TEST(StringHelper, JoinStrings) {
  std::vector<std::string> v;
  v.push_back("hello");
  v.push_back("world");

  std::string result = paddle::string::join_strings(v, ' ');
  EXPECT_EQ(result, "hello world");

  result = paddle::string::join_strings(v, '\n');
  EXPECT_EQ(result, "hello\nworld");

  result = paddle::string::join_strings(v, ',');
  EXPECT_EQ(result, "hello,world");

  result = paddle::string::join_strings(v, " new ");
  EXPECT_EQ(result, "hello new world");
}

TEST(StringHelper, JoinStringsWithConversion) {
  std::vector<int> v = {2, 3};
  auto result =
      paddle::string::join_strings(v, ",", [](int x) { return x * x; });
  EXPECT_EQ(result, "4,9");
}
