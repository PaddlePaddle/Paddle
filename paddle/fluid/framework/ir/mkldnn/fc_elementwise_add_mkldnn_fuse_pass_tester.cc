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

OpDesc* Create_Op_FC(ProgramDesc* prog,
                     const std::vector<test::InOutVarNamePair>& inputs,
                     const std::vector<test::InOutVarNamePair>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType("fc");
  op->SetAttr("use_mkldnn", true);
  op->SetAttr("in_num_col_dims", 1);

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
  Create_Op_FC(&prog, {{"Input", "b"}, {"Bias", "bias"}, {"W", "weights"}},
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
  Create_Op_FC(&prog, {{"Input", "b"}, {"Bias", "bias"}, {"W", "weights"}},
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
  Create_Op_FC(&prog, {{"Input", "b"}, {"Bias", "bias"}, {"W", "weights"}},
               {{"Out", "c"}});

  Create_Op_FC(&prog, {{"Input", "d"}, {"Bias", "bias2"}, {"W", "weights2"}},
               {{"Out", "e"}});

  Create_Op_elementwise_add(&prog, {{"X", "c"}, {"Y", "e"}}, {{"Out", "f"}});
  test::CreateOp(&prog, "relu", {{"X", "f"}}, {{"Out", "g"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(
      &graph, "fc_elementwise_add_mkldnn_fuse_pass", "a", "g", 0, 0));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 2}, {"elementwise_add", 1}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, FC_Residual_VITOCR) {
  auto prog = test::BuildProgramDesc(
      {"a", "b", "c", "d", "e", "f", "g", "h", "i"},
      {"ln_bias", "ln_scale", "bias", "weights", "bias2", "weights2"});

  Create_Op_elementwise_add(&prog, {{"X", "a"}, {"Y", "b"}}, {{"Out", "c"}});

  test::CreateOp(&prog, "layer_norm",
                 {{"X", "c"}, {"Bias", "ln_bias"}, {"Scale", "ln_scale"}},
                 {{"Y", "d"}});
  Create_Op_FC(&prog, {{"Input", "d"}, {"Bias", "bias"}, {"W", "weights"}},
               {{"Out", "e"}});
  test::CreateOp(&prog, "gelu", {{"X", "e"}}, {{"Out", "f"}});
  Create_Op_FC(&prog, {{"Input", "f"}, {"Bias", "bias2"}, {"W", "weights2"}},
               {{"Out", "g"}});
  Create_Op_elementwise_add(&prog, {{"X", "g"}, {"Y", "c"}}, {{"Out", "h"}});
  test::CreateOp(&prog, "relu", {{"X", "h"}}, {{"Out", "i"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "fc_elementwise_add_mkldnn_fuse_pass", "a",
                                     "i", nodes_removed, nodes_added));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 2}, {"elementwise_add", 1}}));
}

TEST(FCElementwiseAddMKLDNNFusePass, FC_Residual_Sequence) {
  auto prog = test::BuildProgramDesc(
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"},
      {"ln_bias", "ln_scale", "bias", "weights", "bias2", "weights2",
       "ln_bias2", "ln_scale2", "bias3", "weights3", "bias4", "weights4"});

  Create_Op_elementwise_add(&prog, {{"X", "a"}, {"Y", "b"}}, {{"Out", "c"}});

  test::CreateOp(&prog, "layer_norm",
                 {{"X", "c"}, {"Bias", "ln_bias"}, {"Scale", "ln_scale"}},
                 {{"Y", "d"}});
  Create_Op_FC(&prog, {{"Input", "d"}, {"Bias", "bias"}, {"W", "weights"}},
               {{"Out", "e"}});
  test::CreateOp(&prog, "gelu", {{"X", "e"}}, {{"Out", "f"}});
  Create_Op_FC(&prog, {{"Input", "f"}, {"Bias", "bias2"}, {"W", "weights2"}},
               {{"Out", "g"}});
  Create_Op_elementwise_add(&prog, {{"X", "g"}, {"Y", "c"}}, {{"Out", "h"}});
  test::CreateOp(&prog, "layer_norm",
                 {{"X", "h"}, {"Bias", "ln_bias2"}, {"Scale", "ln_scale2"}},
                 {{"Y", "i"}});
  Create_Op_FC(&prog, {{"Input", "i"}, {"Bias", "bias3"}, {"W", "weights3"}},
               {{"Out", "j"}});
  test::CreateOp(&prog, "gelu", {{"X", "j"}}, {{"Out", "k"}});
  Create_Op_FC(&prog, {{"Input", "k"}, {"Bias", "bias4"}, {"W", "weights4"}},
               {{"Out", "l"}});
  Create_Op_elementwise_add(&prog, {{"X", "h"}, {"Y", "l"}}, {{"Out", "m"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "fc_elementwise_add_mkldnn_fuse_pass", "a",
                                     "m", nodes_removed * 2, nodes_added * 2));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"fc", 4}, {"elementwise_add", 1}}));
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
