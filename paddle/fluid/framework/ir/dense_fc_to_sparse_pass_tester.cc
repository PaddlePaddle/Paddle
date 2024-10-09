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

#include "paddle/fluid/framework/ir/dense_fc_to_sparse_pass.h"
#include "paddle/fluid/framework/ir/fc_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle::framework::ir {

void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(phi::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "conv2d_filters_0", {});
  AddVarToScope(param_scope, "conv2d_bias_0", {});
  AddVarToScope(param_scope, "weights_0_sparse_2_4", {});
  AddVarToScope(param_scope, "weights_1", {});
  AddVarToScope(param_scope, "bias_1", {});
  AddVarToScope(param_scope, "bias_2", {});
  return param_scope;
}

TEST(FCFusePass, basic) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, filters_0 bias_0)      conv2d           -> conv2d_out
  // conv2d_out                 relu             -> relu_out_0
  // (relu_out_0, weights_0_sparse_2_4)    mul              -> mul_out_0
  // (mul_out_0, bias_1)        elementwise_add  -> add_out_0
  // add_out_0                  relu             -> relu_out_1
  // (relu_out_1, weights_1)    mul              -> mul_out_1
  // (mul_out_1, bias_2)        elementwise_add  -> add_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* filters_0 = layers.data("conv2d_filters_0", {}, true);
  auto* bias_0 = layers.data("conv2d_bias_0", {}, true);
  auto* conv2d_out = layers.conv2d(a, filters_0, bias_0, false);
  auto* relu_out_0 = layers.relu(conv2d_out);
  auto* weights_0 = layers.data("weights_0_sparse_2_4", {5, 4}, true);
  auto* mul_out_0 = layers.mul(relu_out_0, weights_0);
  auto* bias_1 = layers.data("bias_1", {4}, true);
  auto* add_out_0 = layers.elementwise_add(mul_out_0, bias_1, nullptr, 1);
  auto* relu_out_1 = layers.relu(add_out_0);
  auto* weights_1 = layers.data("weights_1", {8, 9}, true);
  auto* mul_out_1 = layers.mul(relu_out_1, weights_1);
  auto* bias_2 = layers.data("bias_2", {1, 9}, true);
  auto* add_out_1 = layers.elementwise_add(mul_out_1, bias_2, nullptr, 1);
  VLOG(4) << add_out_1;

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto fuse_pass = PassRegistry::Instance().Get("fc_fuse_pass");
  auto sparse_pass = PassRegistry::Instance().Get("dense_fc_to_sparse_pass");
  fuse_pass->Set("use_gpu", new bool(true));
  sparse_pass->Set("use_gpu", new bool(true));
  graph->Set("__param_scope__", CreateParamScope());
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  int num_mul_nodes_before = GetNumOpNodes(graph, "mul");
  VLOG(3) << DebugString(graph);

  graph.reset(fuse_pass->Apply(graph.release()));
  graph.reset(sparse_pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_fc_nodes_after = GetNumOpNodes(graph, "fc");
  int num_sparse_fc_nodes_after = GetNumOpNodes(graph, "sparse_fc");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after + 6,
                    common::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before,
                        num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fc_nodes_after,
                    1,
                    common::errors::InvalidArgument("num_fc_nodes_after=%d.",
                                                    num_fc_nodes_after));
  PADDLE_ENFORCE_EQ(num_mul_nodes_before,
                    num_fc_nodes_after + num_sparse_fc_nodes_after,
                    common::errors::InvalidArgument(
                        "num_mul_nodes_before=%d, num_fc_nodes_after=%d + "
                        "num_sparse_fc_nodes_after=%d.",
                        num_mul_nodes_before,
                        num_fc_nodes_after,
                        num_sparse_fc_nodes_after));
}

}  // namespace paddle::framework::ir

USE_PASS(fc_fuse_pass);
USE_PASS(dense_fc_to_sparse_pass);
