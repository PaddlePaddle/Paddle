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

#include "paddle/fluid/framework/ir/conv_bn_fuse_pass.h"

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
  AddVarToScope(param_scope, "bias_1", {3});
  AddVarToScope(param_scope, "scale", {3});
  AddVarToScope(param_scope, "mean", {3});
  AddVarToScope(param_scope, "variance", {3});
  AddVarToScope(param_scope, "filters", {3, 3, 2, 2});
  return param_scope;
}

void TestMain(const std::string& conv_type) {
  // inputs                           operator            output
  // ------------------------------------------------------------------
  // (in, filters, bias_0)            conv           ->   conv_out
  // (conv_out, scale,
  //  bias_1, mean, varaince)         batch_norm     ->   (...)
  Layers layers;
  auto* in = layers.data("in", {1, 3, 20, 20});
  auto* filters = layers.data("filters", {3, 3, 2, 2}, true);
  auto* bias_0 = layers.data("bias_0", {3}, true);
  VarDesc* conv_out;
  if (conv_type == "conv_transpose") {
    conv_out = layers.conv2d_transpose(in, filters, bias_0);
  } else {
    conv_out = layers.conv2d(in, filters, bias_0);
  }
  conv_out->SetShape({1, 3, 20, 20});
  auto* scale = layers.data("scale", {3}, true);
  auto* bias_1 = layers.data("bias_1", {3}, true);
  auto* mean = layers.data("mean", {3}, true);
  auto* variance = layers.data("variance", {3}, true);
  layers.batch_norm(conv_out, scale, bias_1, mean, variance);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());
  auto pass = PassRegistry::Instance().Get(conv_type + "_bn_fuse_pass");
  int num_bn_nodes_before = GetNumOpNodes(graph, "batch_norm");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_bn_nodes_after = GetNumOpNodes(graph, "batch_norm");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_bn_nodes_before, 1);
  PADDLE_ENFORCE_EQ(num_bn_nodes_after, 0);
}

TEST(ConvBNFusePass, conv2d) { TestMain("conv"); }

TEST(ConvBNFusePass, conv2d_tranpose) { TestMain("conv_transpose"); }

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_bn_fuse_pass);
