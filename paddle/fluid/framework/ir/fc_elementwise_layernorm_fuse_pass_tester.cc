/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/fc_elementwise_layernorm_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle::framework::ir {

TEST(FCElementwiseLayerNormFusePass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x, weights_0, bias_0)           fc               -> fc_out_0
  // (fc_out_0, weights_1, bias_1)    fc               -> fc_out_1
  // (fc_out_1, y)                    elementwise_add  -> elementwise_out
  // (elementwise_out, scale, bias_2) layer_norm       ->
  Layers layers;
  auto* x = layers.data("x", {128, 768});
  auto* weights_0 = layers.data("weights_0", {768, 3072}, true);
  auto* bias_0 = layers.data("bias_0", {3072}, true);
  auto* fc_out_0 = layers.fc(x, weights_0, bias_0);  // {128, 3072}
  auto* weights_1 = layers.data("weights_1", {3072, 768}, true);
  auto* bias_1 = layers.data("bias_1", {768}, true);
  auto* fc_out_1 =
      layers.fc(fc_out_0, weights_1, bias_1, 1, "relu");  // {128, 768}
  fc_out_1->SetShape({128, 768});
  auto* y = layers.data("y", {128, 768});
  auto* elementwise_out = layers.elementwise_add(fc_out_1, y);
  auto* scale = layers.data("scale", {768}, true);
  auto* bias_2 = layers.data("bias_2", {768}, true);
  layers.layer_norm(elementwise_out, scale, bias_2);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass =
      PassRegistry::Instance().Get("fc_elementwise_layernorm_fuse_pass");
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_fused_nodes_after =
      GetNumOpNodes(graph, "fused_fc_elementwise_layernorm");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(
      num_nodes_before,
      num_nodes_after + 6,
      common::errors::InvalidArgument(
          "After pass, the number of nodes should be reduced by 6, but the "
          "number before pass is %d, after pass is %d.",
          num_nodes_before,
          num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fused_nodes_after,
                    1,
                    common::errors::InvalidArgument(
                        "After pass, the number of nodes of type "
                        "'fused_fc_elementwise_layernorm' should be 1, not %d.",
                        num_fused_nodes_after));
}

}  // namespace paddle::framework::ir

USE_PASS(fc_elementwise_layernorm_fuse_pass);
