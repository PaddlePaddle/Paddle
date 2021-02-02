/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/scope.h"

#include "gtest/gtest.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

using paddle::framework::Scope;
using paddle::framework::Variable;

TEST(Scope, VarsShadowing) {
  Scope s;
  Scope& ss1 = s.NewScope();
  Scope& ss2 = s.NewScope();

  Variable* v0 = s.Var("a");
  Variable* v1 = ss1.Var("a");

  EXPECT_NE(v0, v1);

  EXPECT_EQ(v0, s.FindVar("a"));
  EXPECT_EQ(v1, ss1.FindVar("a"));
  EXPECT_EQ(v0, ss2.FindVar("a"));
}

TEST(Scope, FindVar) {
  Scope s;
  Scope& ss = s.NewScope();

  EXPECT_EQ(nullptr, s.FindVar("a"));
  EXPECT_EQ(nullptr, ss.FindVar("a"));

  ss.Var("a");

  EXPECT_EQ(nullptr, s.FindVar("a"));
  EXPECT_NE(nullptr, ss.FindVar("a"));
}

TEST(Scope, FindScope) {
  Scope s;
  Scope& ss = s.NewScope();
  Variable* v = s.Var("a");

  EXPECT_EQ(&s, s.FindScope(v));
  EXPECT_EQ(&s, s.FindScope("a"));
  EXPECT_EQ(&s, ss.FindScope(v));
  EXPECT_EQ(&s, ss.FindScope("a"));
}

TEST(Scope, GetAllNames) {
  Scope s;
  Variable* v = s.Var("a");
  EXPECT_EQ(&s, s.FindScope(v));

  std::vector<std::string> ans = s.LocalVarNames();
  std::string str;
  for (auto& var : ans) {
    str += var;
  }

  EXPECT_STREQ("a", str.c_str());
}
