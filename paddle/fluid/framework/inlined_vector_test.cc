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

#include "paddle/fluid/framework/inlined_vector.h"
#include <vector>
#include "gtest/gtest.h"

namespace paddle {
namespace framework {

TEST(inlined_stack, inlined_stack) {
  size_t max_num = 10;

  InlinedVector<size_t, 5> stack;

  for (size_t i = 0; i < max_num; ++i) {
    ASSERT_EQ(stack.size(), i);
    stack.push_back(i);
    ASSERT_EQ(stack.size(), i + 1);
  }

  std::vector<size_t> vec = stack;

  ASSERT_EQ(stack.size(), vec.size());

  for (size_t i = 0; i < vec.size(); ++i) {
    ASSERT_EQ(stack[i], vec[i]);
  }

  for (size_t i = 0; i < max_num; ++i) {
    ASSERT_EQ(stack[i], i);
  }

  for (size_t i = 0; i < max_num; ++i) {
    ASSERT_EQ(stack.back(), max_num - 1 - i);
    stack.pop_back();
    ASSERT_EQ(stack.size(), max_num - 1 - i);
  }
}

}  // namespace framework
}  // namespace paddle
