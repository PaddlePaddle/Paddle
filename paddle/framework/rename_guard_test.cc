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

#include "paddle/framework/rename_guard.h"

#include "gtest/gtest.h"

using paddle::framework::Scope;
using paddle::framework::Variable;
using paddle::framework::RenameGuard;

TEST(RenameGuard, ExchangeVars) {
  Scope s;

  Variable* v0 = s.Var("a");
  Variable* v1 = s.Var("b");

  std::vector<std::pair<std::string, std::string>> var_names = {
      std::make_pair("a", "b")};
  auto* guard = new RenameGuard(s, var_names);

  Variable* v2 = s.FindVar("a");
  Variable* v3 = s.FindVar("b");

  EXPECT_EQ(v0, v3);
  EXPECT_EQ(v1, v2);
  EXPECT_NE(v0, v2);
  EXPECT_NE(v1, v3);

  delete guard;

  v2 = s.FindVar("a");
  v3 = s.FindVar("b");

  EXPECT_EQ(v0, v2);
  EXPECT_EQ(v1, v3);
  EXPECT_NE(v0, v3);
  EXPECT_NE(v1, v2);
}

TEST(RenameGuard, StackedScope) {
  Scope s;
  Scope& ss = s.NewScope();

  Variable* v0 = s.Var("a");
  Variable* v1 = ss.Var("b");

  std::vector<std::pair<std::string, std::string>> var_names = {
      std::make_pair("a", "b")};

  auto* guard = new RenameGuard(ss, var_names);

  Variable* v2 = ss.FindVar("a");
  EXPECT_EQ(v1, v2);
  EXPECT_FALSE(ss.FindVarLocally("b"));

  delete guard;

  v2 = ss.FindVar("a");
  Variable* v3 = ss.FindVar("b");

  EXPECT_EQ(v0, v2);
  EXPECT_EQ(v1, v3);
}
