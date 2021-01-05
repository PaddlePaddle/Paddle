/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include <thread>  // NOLINT
#include <utility>

#include "paddle/fluid/framework/async_variable.h"

#include "gtest/gtest.h"

namespace paddle {
namespace framework {

TEST(AsyncVariableTest, EmplaceGetSameThread) {
  AsyncVariable async_variable;
  int origin = 3;
  async_variable.Emplace<int>(std::forward<int>(origin));
  int value = async_variable.Get<int>();
  EXPECT_EQ(value, 3);

  int* mutable_value = async_variable.GetMutable<int>();
  EXPECT_EQ(*mutable_value, 3);

  *mutable_value = 5;
  EXPECT_EQ(value, 3);
  EXPECT_EQ(async_variable.Get<int>(), 5);
}

void set_async_variable(AsyncVariable* async_variable) {
  async_variable->Emplace<int64_t>(std::forward<int64_t>(57));
}

TEST(AsyncVariableTest, ThreadJoin) {
  AsyncVariable async_variable;
  std::thread check_thread = std::thread(set_async_variable, &async_variable);
  check_thread.join();
  int64_t value = async_variable.Get<int64_t>();
  EXPECT_EQ(value, 57);
}

}  // namespace framework
}  // namespace paddle
