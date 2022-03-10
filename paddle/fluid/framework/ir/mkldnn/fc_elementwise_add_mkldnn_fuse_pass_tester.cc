// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/fc_elementwise_add_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_test_util.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

// Nodes elementwise_add and FC_output are deleted
// FC node is removed and new version with fuse-pass is added
// In general, the graph is 2 vertices smaller (per fuse-pass)
constexpr int nodes_removed = 3;
constexpr int nodes_added = 1;

OpDesc* Create_Op_elementwise_add(
    ProgramDesc* prog, const std::vector<test::InOutVarNamePair>& inputs,
    const std::vector<test::InOutVarNamePair>& outputs,
    bool use_mkldnn = true) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType("elementwise_add");
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("axis", -1);

  for (const auto& input : inputs) {
    op->SetInput(input.first, {input.second});
  }
  for (const auto& output : outputs) {
    op->SetOutput(output.first, {output.second});
  }

  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
  return op;
}

TEST(FCElementwiseAddMKLDNNFusePass, FCBiasAsY) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "fc",
                 {{"Input", "b"}, {"Bias", "bias"}, {"W", "weights"}},
                 {{"Out", "c"}});
  Create_Op_elementwise_add(&prog, {{"X", "a"}, {"Y", "c"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "fc_elementwise_add_mkldnn_fuse_pass", "a",
                                     "e", nodes_removed, nodes_added));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"elementwise_add", 0}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, FCBiasAsX_FCBiasAsY) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e", "f"},
                                     {"bias", "weights", "bias2", "weights2"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});

  // left branch
  test::CreateOp(&prog, "fc",
                 {{"Input", "a"}, {"Bias", "bias"}, {"W", "weights"}},
                 {{"Out", "f"}});

  // right branch
  test::CreateOp(&prog, "fc",
                 {{"Input", "b"}, {"Bias", "bias2"}, {"W", "weights2"}},
                 {{"Out", "c"}});

  Create_Op_elementwise_add(&prog, {{"X", "f"}, {"Y", "c"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "fc_elementwise_add_mkldnn_fuse_pass", "a",
                                     "e", nodes_removed, nodes_added));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 2}, {"elementwise_add", 0}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, FCNoBiasAsY) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "fc", {{"Input", "b"}, {"W", "weights"}},
                 {{"Out", "c"}});
  Create_Op_elementwise_add(&prog, {{"X", "a"}, {"Y", "c"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "fc_elementwise_add_mkldnn_fuse_pass", "a",
                                     "e", nodes_removed, nodes_added));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"elementwise_add", 0}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, FCBiasAsX) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "fc",
                 {{"Input", "b"}, {"Bias", "bias"}, {"W", "weights"}},
                 {{"Out", "c"}});

  Create_Op_elementwise_add(&prog, {{"X", "c"}, {"Y", "a"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "fc_elementwise_add_mkldnn_fuse_pass", "a",
                                     "e", nodes_removed, nodes_added));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"elementwise_add", 0}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, FCNoBiasAsX) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "fc", {{"Input", "b"}, {"W", "weights"}},
                 {{"Out", "c"}});
  Create_Op_elementwise_add(&prog, {{"X", "c"}, {"Y", "a"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "fc_elementwise_add_mkldnn_fuse_pass", "a",
                                     "e", nodes_removed, nodes_added));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 1}, {"elementwise_add", 0}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, NoFusion_NotResidualConnection) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e", "f", "g"},
                                     {"bias", "weights", "bias2", "weights2"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "fc",
                 {{"Input", "b"}, {"Bias", "bias"}, {"W", "weights"}},
                 {{"Out", "c"}});

  test::CreateOp(&prog, "fc",
                 {{"Input", "d"}, {"Bias", "bias2"}, {"W", "weights2"}},
                 {{"Out", "e"}});

  Create_Op_elementwise_add(&prog, {{"X", "c"}, {"Y", "e"}}, {{"Out", "f"}});
  test::CreateOp(&prog, "relu", {{"X", "f"}}, {{"Out", "g"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(
      &graph, "fc_elementwise_add_mkldnn_fuse_pass", "a", "g", 0, 0));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 2}, {"elementwise_add", 1}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("fc_elementwise_add_mkldnn_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_elementwise_add_mkldnn_fuse_pass);
