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

#include "paddle/cinn/hlir/framework/scope.h"

#include <gtest/gtest.h>

namespace cinn {
namespace hlir {
namespace framework {

TEST(Scope, basic) {
  Scope scope;
  auto* var = scope.Var<Tensor>("key");
  auto& tensor = absl::get<Tensor>(*var);
  tensor->Resize(Shape{{3, 1}});
  auto* data = tensor->mutable_data<float>(common::DefaultHostTarget());
  data[0] = 0.f;
  data[1] = 1.f;
  data[2] = 2.f;
}

TEST(ScopeTest, TestEraseVar) {
  Scope scope;
  scope.Var<Tensor>("key");
  ASSERT_NE(scope.FindVar("key"), nullptr);
  scope.EraseVar("key");
  EXPECT_EQ(scope.FindVar("key"), nullptr);
  ASSERT_DEATH(scope.EraseVar("key"), "");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
