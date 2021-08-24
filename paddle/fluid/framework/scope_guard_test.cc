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

#include "paddle/fluid/framework/scope_guard.h"
#include "gtest/gtest.h"

namespace paddle {
namespace framework {

TEST(scope_guard, scope_guard_test) {
  int n = 10;
  {
    DEFINE_PADDLE_SCOPE_GUARD([&n] { ++n; });
  }
  EXPECT_EQ(n, 11);
  try {
    DEFINE_PADDLE_SCOPE_GUARD([&] { --n; });
    DEFINE_PADDLE_SCOPE_GUARD([&] { --n; });
    throw std::runtime_error("any exception");
  } catch (std::runtime_error &) {
    EXPECT_EQ(n, 9);
  }
}

}  // namespace framework
}  // namespace paddle
