// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/fc_act_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_test_util.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {
namespace ir {

// ------------------------------ Test cases -----------------------------------

TEST(FuseFCActOneDNNPass, ThrowUseMkldnn) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}}, false);
  test::CreateOp(&prog, "gelu", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  // No fusion in this attribute configuration
  constexpr int removed_nodes_count = 0;

  EXPECT_THROW(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                      "act_y", removed_nodes_count),
               paddle::platform::EnforceNotMet);
}

TEST(FuseFCActOneDNNPass, FuseWithGeluTanh) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}});
  auto* act_op = test::CreateOp(&prog, "gelu", {{"Input", "fc_y"}},
                                {{"Out", "act_y"}}, false);
  act_op->SetAttr("approximate", true);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"gelu", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_EQ(act_type.compare("gelu_tanh"), 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithGeluErf) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}});
  auto* act_op = test::CreateOp(&prog, "gelu", {{"Input", "fc_y"}},
                                {{"Out", "act_y"}}, false);
  act_op->SetAttr("approximate", false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"gelu", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_EQ(act_type.compare("gelu_erf"), 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithGeluAuto) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}});
  test::CreateOp(&prog, "gelu", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"gelu", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_EQ(act_type.compare("gelu"), 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithTanh) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}});
  test::CreateOp(&prog, "tanh", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"tanh", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_EQ(act_type.compare("tanh"), 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithSigmoid) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}});
  test::CreateOp(&prog, "sigmoid", {{"Input", "fc_y"}}, {{"Out", "act_y"}},
                 false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"sigmoid", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_EQ(act_type.compare("sigmoid"), 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithMish) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}});
  test::CreateOp(&prog, "mish", {{"Input", "fc_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"mish", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_EQ(act_type.compare("mish"), 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, FuseWithHardSwish) {
  auto prog =
      test::BuildProgramDesc({"x", "fc_y", "act_y"}, {"weights", "bias"});
  test::CreateOp(&prog, "fc",
                 {
                     {"Input", "x"}, {"Weights", "weights"}, {"Bias", "bias"},
                 },
                 {{"Out", "fc_y"}});
  test::CreateOp(&prog, "hard_swish", {{"Input", "fc_y"}}, {{"Out", "act_y"}},
                 false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "fc_act_mkldnn_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"hard_swish", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("activation_type"));
      auto act_type =
          BOOST_GET_CONST(std::string, op->GetAttr("activation_type"));
      EXPECT_EQ(act_type.compare("hard_swish"), 0);
    }
  }
}

TEST(FuseFCActOneDNNPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("fc_act_mkldnn_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_act_mkldnn_fuse_pass);
