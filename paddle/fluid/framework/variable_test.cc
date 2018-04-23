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

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/variable.h"

TEST(Variable, GetMutable) {
  using paddle::framework::Variable;

  struct Tensor {
    int content_;
  };

  std::unique_ptr<Variable> v(new Variable());

  Tensor* t = v->GetMutable<Tensor>();
  t->content_ = 1234;

  const Tensor& tt = v->Get<Tensor>();
  EXPECT_EQ(1234, tt.content_);

  std::string* s = v->GetMutable<std::string>();
  *s = "hello";

  const std::string& ss = v->Get<std::string>();
  EXPECT_EQ("hello", ss);
}
