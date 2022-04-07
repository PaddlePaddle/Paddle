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

#include "paddle/fluid/framework/ir/mkldnn/reshape_transpose_matmul_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/mkldnn/reshape_transpose_matmul_v2_mkldnn_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void AddVarToScope(Scope* param_scope, const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<LoDTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(platform::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "w1", {768, 768});
  AddVarToScope(param_scope, "bias1", {768});
  AddVarToScope(param_scope, "w2", {768, 768});
  AddVarToScope(param_scope, "bias2", {768});
  return param_scope;
}

void TestMain(const std::string& op_name, bool with_xshapes) {
  // inputs          operator          output
  // -----------------------------------------------
  //  a1,w1,bias1      fc          ->    b1
  //  b1             reshape       ->    c1
  //  c1            transpose      ->    d1
  //  a2,w2,bias2      fc          ->    b2
  //  b2             reshape       ->    c2
  //  c2            transpose      ->    d2
  // (d1, d2)        matmul(_v2)   ->    (...)
  Layers layers;
  auto* a1 = layers.data("a1", {-1, 128, 768});
  auto* w1 = layers.data("w1", {768, 768}, true);
  auto* bias1 = layers.data("bias1", {768}, true);
  auto* b1 = layers.fc(a1, w1, bias1, 2);
  b1->SetShape({-1, 128, 768});
  auto* c1 = layers.reshape2(b1, {0, 0, 12, 64}, with_xshapes);
  c1->SetShape({-1, 128, 12, 64});
  auto* d1 = layers.transpose2(c1, {0, 2, 1, 3}, with_xshapes);
  d1->SetShape({-1, 12, 128, 64});
  auto* a2 = layers.data("a2", {-1, 128, 768});
  auto* w2 = layers.data("w2", {768, 768}, true);
  auto* bias2 = layers.data("bias2", {768}, true);
  auto* b2 = layers.fc(a2, w2, bias2, 2);
  b2->SetShape({-1, 128, 768});
  auto* c2 = layers.reshape2(b2, {0, 0, 12, 64});
  c2->SetShape({-1, 128, 12, 64});
  auto* d2 = layers.transpose2(c2, {0, 2, 1, 3});
  d2->SetShape({-1, 12, 128, 64});
  if (op_name == "matmul_v2") {
    layers.matmul_v2(d1, d2);
  } else {
    layers.matmul(d1, d2);
  }

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());

  int num_reshape_nodes_before = GetNumOpNodes(graph, "reshape2");
  int num_transpose_nodes_before = GetNumOpNodes(graph, "transpose2");
  int total_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  auto pass = PassRegistry::Instance().Get("reshape_transpose_" + op_name +
                                           "_mkldnn_fuse_pass");
  graph.reset(pass->Apply(graph.release()));

  int num_reshape_nodes_after = GetNumOpNodes(graph, "reshape2");
  int num_transpose_nodes_after = GetNumOpNodes(graph, "transpose2");
  int total_nodes_after = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  EXPECT_EQ(num_reshape_nodes_before, 2);
  EXPECT_EQ(num_reshape_nodes_after, 0);
  EXPECT_EQ(num_transpose_nodes_before, 2);
  EXPECT_EQ(num_transpose_nodes_after, 0);
  int removed = 8;  // 2* reshape, reshape_out, transpose, transpose_out
  if (with_xshapes) removed += 2;  // transpose_xshape, reshape_xshape
  EXPECT_EQ(total_nodes_before - removed, total_nodes_after);
  auto* matmul_op_desc = GetOpNodes(graph, op_name).at(0)->Op();

  auto check = [&matmul_op_desc](std::string a) {
    std::string shape_str = "fused_reshape_" + a;
    auto shape = matmul_op_desc->GetAttrIfExists<std::vector<int>>(shape_str);
    EXPECT_EQ(shape, (std::vector<int>{0, 0, 12, 64}));
    std::string axis_str = "fused_transpose_" + a;
    auto axis = matmul_op_desc->GetAttrIfExists<std::vector<int>>(axis_str);
    EXPECT_EQ(axis, (std::vector<int>{0, 2, 1, 3}));
  };
  check("X");
  check("Y");
}

TEST(ReshapeTransposeMatmulMkldnnFusePass,
     both_matmul_inputs_reshape_transpose) {
  TestMain("matmul", false);
}

TEST(ReshapeTransposeMatmulMkldnnFusePass,
     both_matmul_inputs_reshape_transpose_one_with_xshapes) {
  TestMain("matmul", true);
}

TEST(ReshapeTransposeMatmulV2MkldnnFusePass,
     both_matmulv2_inputs_reshape_transpose) {
  TestMain("matmul_v2", false);
}

TEST(ReshapeTransposeMatmulV2MkldnnFusePass,
     both_matmulv2_inputs_reshape_transpose_one_with_xshapes) {
  TestMain("matmul_v2", true);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(reshape_transpose_matmul_mkldnn_fuse_pass);
USE_PASS(reshape_transpose_matmul_v2_mkldnn_fuse_pass);
