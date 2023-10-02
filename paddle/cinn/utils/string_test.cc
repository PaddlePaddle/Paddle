// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/utils/string.h"

#include <gtest/gtest.h>

namespace cinn {
namespace utils {

TEST(string, Endswith) {
  std::string a = "a__p";
  ASSERT_TRUE(Endswith(a, "__p"));
  ASSERT_FALSE(Endswith(a, "_x"));
  ASSERT_TRUE(Endswith(a, "a__p"));
  ASSERT_FALSE(Endswith(a, "a___p"));
}
TEST(string, Startswith) {
  std::string a = "a__p";
  ASSERT_TRUE(Startswith(a, "a_"));
  ASSERT_TRUE(Startswith(a, "a__"));
  ASSERT_TRUE(Startswith(a, "a__p"));
  ASSERT_FALSE(Startswith(a, "a___p"));
}

}  // namespace utils
}  // namespace cinn
