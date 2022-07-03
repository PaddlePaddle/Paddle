// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_elementwise_add_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_test_util.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

constexpr int nodes_removed = 3;
constexpr int nodes_added = 1;

OpDesc* Create_Op_con2d(ProgramDesc* prog,
                        const std::string& op_type_name,
                        const std::vector<test::InOutVarNamePair>& inputs,
                        const std::vector<test::InOutVarNamePair>& outputs,
                        const bool use_mkldnn = true) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  const std::vector<int> strides({1, 1});
  const std::vector<int> paddings({0, 0});
  const std::vector<int> dilations({1, 1});
  op->SetType(op_type_name);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("strides", strides);
  op->SetAttr("groups", 1);
  op->SetAttr("paddings", paddings);
  op->SetAttr("padding_algorithm", std::string("EXPLICIT"));
  op->SetAttr("dilations", dilations);
  op->SetAttr("data_format", std::string("NCHW"));

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

OpDesc* Create_Op_elemntwise_add(
    ProgramDesc* prog,
    const std::string& op_type_name,
    const std::vector<test::InOutVarNamePair>& inputs,
    const std::vector<test::InOutVarNamePair>& outputs,
    bool use_mkldnn = true) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(op_type_name);
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

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionAsYWithElementwiseAddRelu) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
                  {{"Output", "c"}});
  Create_Op_elemntwise_add(
      &prog, "elementwise_add", {{"X", "a"}, {"Y", "c"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a",
                                     "relu",
                                     nodes_removed,
                                     nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionProjectionAsYWithElementwiseAddRelu) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e", "f"},
                                     {"bias", "weights", "bias2", "weights2"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  // right branch
  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
                  {{"Output", "c"}});

  // left branch
  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "a"}, {"Bias", "bias2"}, {"Filter", "weights2"}},
                  {{"Output", "f"}});

  Create_Op_elemntwise_add(
      &prog, "elementwise_add", {{"X", "f"}, {"Y", "c"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a",
                                     "relu",
                                     nodes_removed,
                                     nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 2}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionAsYWithElementwiseAddReluNoBias) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "b"}, {"Filter", "weights"}},
                  {{"Output", "c"}});
  Create_Op_elemntwise_add(
      &prog, "elementwise_add", {{"X", "a"}, {"Y", "c"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a",
                                     "relu",
                                     nodes_removed,
                                     nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionAsXWithElementwiseAddRelu) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
                  {{"Output", "c"}});

  Create_Op_elemntwise_add(
      &prog, "elementwise_add", {{"X", "c"}, {"Y", "a"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a",
                                     "relu",
                                     nodes_removed,
                                     nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionAsXWithElementwiseAddReluNoBias) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "b"}, {"Filter", "weights"}},
                  {{"Output", "c"}});
  Create_Op_elemntwise_add(
      &prog, "elementwise_add", {{"X", "c"}, {"Y", "a"}}, {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a",
                                     "relu",
                                     nodes_removed,
                                     nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass, NoFusion) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e", "f", "g"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "b"}, {"Filter", "weights"}},
                  {{"Output", "c"}});

  Create_Op_con2d(&prog,
                  "conv2d",
                  {{"Input", "d"}, {"Filter", "weights"}},
                  {{"Output", "e"}});

  Create_Op_elemntwise_add(
      &prog, "elementwise_add", {{"X", "c"}, {"Y", "e"}}, {{"Out", "f"}});
  test::CreateOp(&prog, "relu", {{"X", "f"}}, {{"Out", "g"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(
      &graph, "conv_elementwise_add_mkldnn_fuse_pass", "a", "g", 0, 0));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 2}, {"elementwise_add", 1}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("conv_elementwise_add_mkldnn_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_elementwise_add_mkldnn_fuse_pass);
