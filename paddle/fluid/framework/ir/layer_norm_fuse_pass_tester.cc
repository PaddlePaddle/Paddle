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

#include <gtest/gtest.h>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/layer_norm_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_test_util.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {

ProgramDesc BuildGraphProgram() {
  auto prog = test::BuildProgramDesc(
      {"x", "x_mean_out", "x_sub_mean_out", "x_sub_mean_sqr_out", "std_dev_out",
       "std_dev_eps_out", "std_dev_eps_sqrt_out", "division_out", "scale_out",
       "shift_out"},
      {"sqr_pow", "eps", "gamma", "beta"});

  auto* block_desc = prog.MutableBlock(0);
  auto* x_var_desc = block_desc->Var("x");
  x_var_desc->SetDataType(proto::VarType::FP32);
  x_var_desc->SetShape({3, 32, 48});

  auto* eps_var_desc = block_desc->Var("eps");
  eps_var_desc->SetDataType(proto::VarType::FP32);
  eps_var_desc->SetShape({1});

  auto* gamma_var_desc = block_desc->Var("gamma");
  gamma_var_desc->SetDataType(proto::VarType::FP32);
  gamma_var_desc->SetShape({48});

  auto* beta_var_desc = block_desc->Var("beta");
  beta_var_desc->SetDataType(proto::VarType::FP32);
  beta_var_desc->SetShape({48});

  test::CreateOp(&prog, "reduce_mean", {{"X", "x"}}, {{"Out", "x_mean_out"}},
                 false);
  test::CreateOp(&prog, "elementwise_sub", {{"X", "x"}, {"Y", "x_mean_out"}},
                 {{"Out", "x_sub_mean_out"}}, false);
  test::CreateOp(&prog, "elementwise_pow",
                 {{"X", "x_sub_mean_out"}, {"Y", "sqr_pow"}},
                 {{"Out", "x_sub_mean_sqr_out"}}, false);
  test::CreateOp(&prog, "reduce_mean", {{"X", "x_sub_mean_sqr_out"}},
                 {{"Out", "std_dev_out"}}, false);
  test::CreateOp(&prog, "elementwise_add", {{"X", "std_dev_out"}, {"Y", "eps"}},
                 {{"Out", "std_dev_eps_out"}}, false);
  test::CreateOp(&prog, "sqrt", {{"X", "std_dev_eps_out"}},
                 {{"Out", "std_dev_eps_sqrt_out"}}, false);
  test::CreateOp(&prog, "elementwise_div",
                 {{"X", "x_sub_mean_out"}, {"Y", "std_dev_eps_sqrt_out"}},
                 {{"Out", "division_out"}}, false);
  test::CreateOp(&prog, "elementwise_mul",
                 {{"X", "division_out"}, {"Y", "gamma"}},
                 {{"Out", "scale_out"}}, false);
  test::CreateOp(&prog, "elementwise_add", {{"X", "scale_out"}, {"Y", "beta"}},
                 {{"Out", "shift_out"}}, false);
  return prog;
}

bool CheckFusedSubgraphOpsCount(const Graph& graph) {
  return test::AssertOpsCount(graph, {{"reduce_mean", 0},
                                      {"elementwise_sub", 0},
                                      {"elementwise_pow", 0},
                                      {"elementwise_add", 0},
                                      {"sqrt", 0},
                                      {"elementwise_div", 0},
                                      {"elementwise_mul", 0},
                                      {"layer_norm", 1}});
}

}  // namespace

// ------------------------------ Test cases -----------------------------------

TEST(FuseLayerNormPass, TestFuse) {
  ProgramDesc prog = BuildGraphProgram();

  Graph graph(prog);
  constexpr int removed_nodes = 19;
  constexpr int added_nodes = 1;

  auto place = paddle::platform::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  float eps_value = 1e-5f;
  // Init scope, as it is used in pass
  exe.CreateVariables(prog, 0, true, &scope);
  test::InitLoDTensorHolder<float>(&scope, place, "eps", {1}, &eps_value);

  graph.SetNotOwned(kParamScopeAttr, &scope);
  EXPECT_TRUE(test::RunPassAndAssert(&graph, "layer_norm_fuse_pass", "x",
                                     "shift_out", removed_nodes, added_nodes));
  EXPECT_TRUE(CheckFusedSubgraphOpsCount(graph));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "layer_norm") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("is_test"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("is_test")));
      ASSERT_TRUE(op->HasAttr("begin_norm_axis"));
      ASSERT_TRUE(op->HasAttr("epsilon"));
    }
  }
}

TEST(FuseLayerNormPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("layer_norm_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(layer_norm_fuse_pass);
