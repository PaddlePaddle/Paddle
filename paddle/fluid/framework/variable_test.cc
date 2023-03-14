//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/variable.h"

#include "gtest/gtest.h"

namespace paddle {
namespace framework {

TEST(Variable, GetMutable) {
  std::unique_ptr<Variable> v(new Variable());

  auto* t = v->GetMutable<String>();
  *t = "1234";

  const auto& tt = v->Get<String>();
  EXPECT_EQ("1234", tt);

  try {
    v->GetMutable<phi::DenseTensor>();
  } catch (std::exception& e) {
    return;
  }
  EXPECT_TRUE(false);

  std::unique_ptr<Variable> v_ints(new Variable());
  auto* v_t = v_ints->GetMutable<std::vector<int>>();
  v_t->push_back(1);
  v_t->push_back(2);

  const auto& cv_t = v_ints->Get<std::vector<int>>();
  EXPECT_EQ(cv_t[0], 1);
  EXPECT_EQ(cv_t[1], 2);
}

}  // namespace framework
}  // namespace paddle
