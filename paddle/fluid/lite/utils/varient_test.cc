// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/utils/varient.h"
#include <gtest/gtest.h>
#include <set>
#include <string>
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {
namespace utils {

TEST(varient, test) {
  variant<int, float> a;
  // The initial state should be invalid.
  ASSERT_FALSE(a.valid());
  a.set<int>(1);
  ASSERT_EQ(a.get<int>(), 1);
  a.set<int>(20);
  ASSERT_EQ(a.get<int>(), 20);
}

TEST(varient, reference) {
  variant<int, float, std::string> a;
  a.set<std::string>("hello world");

  auto& b = a.get<std::string>();
  ASSERT_EQ(b, "hello world");
}

TEST(varient, get_wrong_type) {
  variant<int, float> a;
  a.set<int>(100);
  bool exception = false;
  try {
    float b = a.get<float>();
    LOG(INFO) << b + 1;
  } catch (...) {
    exception = true;
  }
  ASSERT_TRUE(exception);
}

}  // namespace utils
}  // namespace lite
}  // namespace paddle
