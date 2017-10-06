/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/string/to_string.h"
#include <gtest/gtest.h>

constexpr char kOutputString[] = "User Defined Output";
class UserDefinedClass {
 public:
};

std::ostream& operator<<(std::ostream& s, const UserDefinedClass& ins) {
  s << kOutputString;
  return s;
}

TEST(to_string, normal) {
  using namespace paddle::string;
  ASSERT_EQ("10", to_string(10));
  ASSERT_EQ("abc", to_string("abc"));
  ASSERT_EQ("1.2", to_string(1.2));
}

TEST(to_string, user_defined) {
  using namespace paddle::string;
  UserDefinedClass instance;
  ASSERT_EQ(kOutputString, to_string(instance));
}
