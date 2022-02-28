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

#include "paddle/infrt/host_context/value.h"

#include <gtest/gtest.h>

namespace infrt {
namespace host_context {

TEST(ValueRef, test) {
  ValueRef x(12);
  ASSERT_EQ(x.get<int>(), 12);

  ValueRef y(1.2f);
  ASSERT_EQ(y.get<float>(), 1.2f);

  ValueRef z(true);
  ASSERT_EQ(z.get<bool>(), true);
}

// If the value is not assign, the get_or_default should return a default value.
TEST(Value, init) {
  Value x;
  ASSERT_EQ(x.get_or_default<int>(), 0);

  Value tensor;
  auto& t = tensor.get_or_default<tensor::DenseHostTensor>();
  ASSERT_EQ(t.shape().GetRank(), 0);
}

}  // namespace host_context
}  // namespace infrt
