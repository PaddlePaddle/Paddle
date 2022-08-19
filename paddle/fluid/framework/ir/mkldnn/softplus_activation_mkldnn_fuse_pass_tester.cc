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

#include <vector>

#include "paddle/fluid/framework/ir/mkldnn/softplus_activation_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void MainTest(const std::string& activation_type) {
  auto prog =
      test::BuildProgramDesc({"softplus_x", "softplus_out", "activation_out"});
  test::CreateOp(
      &prog, "softplus", {{"X", "softplus_x"}}, {{"Out", "softplus_out"}});
  test::CreateOp(&prog,
                 activation_type,
                 {{"X", "softplus_out"}},
                 {{"Out", "activation_out"}},
                 false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph,
                                     "softplus_activation_mkldnn_fuse_pass",
                                     "softplus_x",
                                     "activation_out",
                                     removed_nodes_count));
  EXPECT_TRUE(
      test::AssertOpsCount(graph, {{"softplus", 1}, {activation_type, 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "softplus") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(PADDLE_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("fuse_activation"));
      auto activation_type =
          PADDLE_GET_CONST(std::string, op->GetAttr("fuse_activation"));
      EXPECT_EQ(activation_type.compare(activation_type), 0);
    }
  }
}

// clang-format off
TEST(FuseSoftplusActivationOneDNNPass, FuseSoftplusWithTanh) {MainTest("tanh")}

TEST(FuseSoftplusActivationOneDNNPass, FuseSoftplusWithRelu) {MainTest("relu")}

TEST(FuseSoftplusActivationOneDNNPass,
     FuseSoftplusWithLeakyRelu) {MainTest("leaky_relu")}

TEST(FuseSoftplusActivationOneDNNPass,
     FuseSoftplusWithSwish) {MainTest("swish")}

TEST(FuseSoftplusActivationOneDNNPass,
     FuseSoftplusWithHardswish) {MainTest("hardswish")}

TEST(FuseSoftplusActivationOneDNNPass, FuseSoftplusWithSqrt) {MainTest("sqrt")}

TEST(FuseSoftplusActivationOneDNNPass, FuseSoftplusWithAbs) {MainTest("abs")}

TEST(FuseSoftplusActivationOneDNNPass, FuseSoftplusWithClip) {MainTest("clip")}

TEST(FuseSoftplusActivationOneDNNPass, FuseSoftplusWithGelu) {MainTest("gelu")}

TEST(FuseSoftplusActivationOneDNNPass,
     FuseSoftplusWithRelu6) {MainTest("relu6")}

TEST(FuseSoftplusActivationOneDNNPass, FuseSoftplusWithSigmoid) {
  MainTest("sigmoid")
}
// clang-format on

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(softplus_activation_mkldnn_fuse_pass);
