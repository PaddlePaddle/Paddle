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

#include "paddle/fluid/framework/ir/fc_gru_fuse_pass.h"

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
  AddVarToScope(param_scope, "gru_fc_w", {});
  AddVarToScope(param_scope, "gru_fc_b", {});
  AddVarToScope(param_scope, "gru_w", {});
  AddVarToScope(param_scope, "gru_b", {});
  AddVarToScope(param_scope, "gru_batch_gate_0", {});
  AddVarToScope(param_scope, "gru_batch_reset_hidden_prev_0", {});
  AddVarToScope(param_scope, "gru_batch_hidden_0", {});
  AddVarToScope(param_scope, "gru_hidden_0", {});
  AddVarToScope(param_scope, "gru_batch_gate_1", {});
  AddVarToScope(param_scope, "gru_batch_reset_hidden_prev_1", {});
  AddVarToScope(param_scope, "gru_batch_hidden_1", {});
  AddVarToScope(param_scope, "gru_hidden_1", {});
  return param_scope;
}

TEST(FCFusePass, basic) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, gru_fc_w)                mul         ->   fc_0_tmp_0
  // (fc_0_tmp_0, gru_fc_b)  elementwise_add  ->   fc_0_tmp_1
  // (fc_0_tmp_1,gru_w,gru_b      gru         ->   gru_out_0

  // (b, gru_fc_w)                mul         ->   fc_1_tmp_0
  // (fc_1_tmp_0, gru_fc_b)  elementwise_add  ->   fc_1_tmp_1
  // (fc_1_tmp_1,gru_w,gru_b)     gru         ->   gru_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* b = layers.data("b");
  auto* fc_w = layers.data("gru_fc_w", {}, true);
  auto* fc_b = layers.data("gru_fc_b", {}, true);
  auto* gru_w = layers.data("gru_w", {}, true);
  auto* gru_b = layers.data("gru_b", {}, true);
  auto* fc_0_tmp0 = layers.mul(a, fc_w);
  auto* fc_0_tmp1 = layers.elementwise_add(fc_0_tmp0, fc_b);
  auto* gru_batch_gate_0 = layers.data("gru_batch_gate_0", {}, false);
  auto* gru_batch_reset_hidden_prev_0 =
      layers.data("gru_batch_reset_hidden_prev_0", {}, false);
  auto* gru_batch_hidden_0 = layers.data("gru_batch_hidden_0", {}, false);
  auto* gru_hidden_0 = layers.data("gru_hidden_0", {}, false);
  layers.gru(fc_0_tmp1, gru_w, gru_b, gru_batch_gate_0,
             gru_batch_reset_hidden_prev_0, gru_batch_hidden_0, gru_hidden_0);

  auto* fc_1_tmp0 = layers.mul(b, fc_w);
  auto* fc_1_tmp1 = layers.elementwise_add(fc_1_tmp0, fc_b);
  auto* gru_batch_gate_1 = layers.data("gru_batch_gate_1", {}, false);
  auto* gru_batch_reset_hidden_prev_1 =
      layers.data("gru_batch_reset_hidden_prev_1", {}, false);
  auto* gru_batch_hidden_1 = layers.data("gru_batch_hidden_1", {}, false);
  auto* gru_hidden_1 = layers.data("gru_hidden_1", {}, false);
  layers.gru(fc_1_tmp1, gru_w, gru_b, gru_batch_gate_1,
             gru_batch_reset_hidden_prev_1, gru_batch_hidden_1, gru_hidden_1);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("fc_gru_fuse_pass");
  pass->Set("use_gpu", new bool(true));
  graph->Set("__param_scope__", CreateParamScope());
  int num_nodes_before = graph->Nodes().size();
  int num_gru_nodes_before = GetNumOpNodes(graph, "gru");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fuse_gru_nodes_after = GetNumOpNodes(graph, "fusion_gru");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 6,
                    platform::errors::PreconditionNotMet(
                        "The number of nodes before and after "
                        "the fuse does not meet expectations"));
  PADDLE_ENFORCE_EQ(
      num_fuse_gru_nodes_after, 2,
      platform::errors::PreconditionNotMet("The number of gru nodes before the "
                                           "fuse does not meet expectations"));
  PADDLE_ENFORCE_EQ(num_gru_nodes_before, num_fuse_gru_nodes_after,
                    platform::errors::PreconditionNotMet(
                        "The number of fusion_gru nodes does not meet "
                        "expectations after fuse"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_gru_fuse_pass);
