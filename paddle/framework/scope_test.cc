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

#include "paddle/framework/scope.h"
#include "gtest/gtest.h"

TEST(Scope, Create) {
  using paddle::framework::Scope;
  using paddle::framework::Variable;

  Scope* scope = new Scope();

  Variable* var0 = scope->CreateVariable("");
  EXPECT_NE(var0, nullptr);

  Variable* var1 = scope->GetVariable("a");
  EXPECT_EQ(var1, nullptr);

  Variable* var2 = scope->CreateVariable("a");

  ASSERT_DEATH({ scope->CreateVariable("a"); }, "");

  Variable* var3 = scope->GetVariable("a");
  EXPECT_EQ(var2, var3);

  Variable* var4 = scope->GetOrCreateVariable("a");
  EXPECT_EQ(var2, var4);
}

TEST(Scope, Parent) {
  using paddle::framework::Scope;
  using paddle::framework::Variable;

  const auto parent_scope_ptr = std::shared_ptr<Scope>(new Scope());
  Scope* scope = new Scope(parent_scope_ptr);

  Variable* var0 = parent_scope_ptr->CreateVariable("a");
  EXPECT_NE(var0, nullptr);

  Variable* var1 = scope->GetVariable("a");
  EXPECT_EQ(var0, var1);
}
