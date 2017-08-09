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

constexpr char OUT_STR[] = "User Defined Output";
class UserDefinedClass {
public:
};

std::ostream& operator<<(std::ostream& s, const UserDefinedClass& ins) {
  s << OUT_STR;
  return s;
}

// android macro comes from
// https://stackoverflow.com/questions/15328751/android-macro-suddenly-not-defined
#if !defined(ANDROID) && !defined(__ANDROID__)
// In android, std::to_string is not defined.
// https://stackoverflow.com/questions/22774009/android-ndk-stdto-string-support
TEST(to_string, normal) {
  using namespace paddle::string;
  ASSERT_EQ(std::to_string(10), to_string(10));
  ASSERT_EQ("abc", to_string("abc"));

  auto std_to_string = std::to_string(1.2);
  auto my_to_string = to_string(1.2);

  // std::to_string might fill zero after float value, like 1.2000
  for (size_t i = 0; i < my_to_string.size(); ++i) {
    ASSERT_EQ(my_to_string[i], std_to_string[i]);
  }
}
#endif

TEST(to_string, user_defined) {
  using namespace paddle::string;
  UserDefinedClass instance;
  ASSERT_EQ(OUT_STR, to_string(instance));
}