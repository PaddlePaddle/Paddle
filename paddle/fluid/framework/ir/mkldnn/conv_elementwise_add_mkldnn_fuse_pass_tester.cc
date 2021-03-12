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
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

constexpr int nodes_removed = 3;
constexpr int nodes_added = 1;

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionAsYWithElementwiseAddRelu) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "conv2d",
                 {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
                 {{"Output", "c"}});
  test::CreateOp(&prog, "elementwise_add", {{"X", "a"}, {"Y", "c"}},
                 {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a", "relu", nodes_removed, nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionProjectionAsYWithElementwiseAddRelu) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e", "f"},
                                     {"bias", "weights", "bias2", "weights2"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  // right branch
  test::CreateOp(&prog, "conv2d",
                 {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
                 {{"Output", "c"}});

  // left branch
  test::CreateOp(&prog, "conv2d",
                 {{"Input", "a"}, {"Bias", "bias2"}, {"Filter", "weights2"}},
                 {{"Output", "f"}});

  test::CreateOp(&prog, "elementwise_add", {{"X", "f"}, {"Y", "c"}},
                 {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a", "relu", nodes_removed, nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 2}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionAsYWithElementwiseAddReluNoBias) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "conv2d", {{"Input", "b"}, {"Filter", "weights"}},
                 {{"Output", "c"}});
  test::CreateOp(&prog, "elementwise_add", {{"X", "a"}, {"Y", "c"}},
                 {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a", "relu", nodes_removed, nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionAsXWithElementwiseAddRelu) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"bias", "weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "conv2d",
                 {{"Input", "b"}, {"Bias", "bias"}, {"Filter", "weights"}},
                 {{"Output", "c"}});

  test::CreateOp(&prog, "elementwise_add", {{"X", "c"}, {"Y", "a"}},
                 {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a", "relu", nodes_removed, nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass,
     ConvolutionAsXWithElementwiseAddReluNoBias) {
  auto prog = test::BuildProgramDesc({"a", "b", "c", "d", "e"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "conv2d", {{"Input", "b"}, {"Filter", "weights"}},
                 {{"Output", "c"}});
  test::CreateOp(&prog, "elementwise_add", {{"X", "c"}, {"Y", "a"}},
                 {{"Out", "d"}});
  test::CreateOp(&prog, "relu", {{"X", "d"}}, {{"Out", "e"}});

  Graph graph(prog);

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "conv_elementwise_add_mkldnn_fuse_pass",
                                     "a", "relu", nodes_removed, nodes_added));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"conv2d", 1}, {"elementwise_add", 0}}));
}

TEST(ConvElementwiseAddMKLDNNFusePass, NoFusion) {
  auto prog =
      test::BuildProgramDesc({"a", "b", "c", "d", "e", "f", "g"}, {"weights"});

  test::CreateOp(&prog, "sigmoid", {{"X", "a"}}, {{"Out", "b"}});
  test::CreateOp(&prog, "conv2d", {{"Input", "b"}, {"Filter", "weights"}},
                 {{"Output", "c"}});

  test::CreateOp(&prog, "conv2d", {{"Input", "d"}, {"Filter", "weights"}},
                 {{"Output", "e"}});

  test::CreateOp(&prog, "elementwise_add", {{"X", "c"}, {"Y", "e"}},
                 {{"Out", "f"}});
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
