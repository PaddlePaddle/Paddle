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

  auto scope = std::make_shared<Scope>();

  Variable* var0 = scope->CreateVariable("");
  EXPECT_NE(var0, nullptr);

  /// GetVariable will return nullptr if not exist.
  Variable* var1 = scope->GetVariable("a");
  EXPECT_EQ(var1, nullptr);

  /// CreateVariable will return one.
  Variable* var2 = scope->CreateVariable("a");
  EXPECT_NE(var2, nullptr);

  /// Get the created variable.
  Variable* var3 = scope->GetVariable("a");
  EXPECT_EQ(var2, var3);

  /// CreateVariable will just return the variable if it's
  /// already exist.
  Variable* var4 = scope->CreateVariable("a");
  EXPECT_EQ(var4, var2);
}

TEST(Scope, Parent) {
  using paddle::framework::Scope;
  using paddle::framework::Variable;

  auto parent_scope = std::make_shared<Scope>();
  auto scope = std::make_shared<Scope>(parent_scope);

  Variable* var0 = parent_scope->CreateVariable("a");
  EXPECT_NE(var0, nullptr);

  /// GetVariable will get Variable from parent scope if exist.
  Variable* var1 = scope->GetVariable("a");
  EXPECT_EQ(var0, var1);
}
