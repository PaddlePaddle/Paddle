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

#include "paddle/fluid/framework/ir/skip_layernorm_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(SkipLayerNormFusePass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x, y)                       elementwise_add    -> elementwise_out
  // (elementwise_out, scale, bias) layer_norm       -> layer_norm_out...
  Layers layers;
  auto* x = layers.data("x", {128, 768});
  auto* y = layers.data("y", {128, 768});
  auto* elementwise_out = layers.elementwise_add(x, y);
  auto* scale = layers.data("scale", {768}, true);
  auto* bias = layers.data("bias", {768}, true);
  layers.layer_norm(elementwise_out, scale, bias);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("skip_layernorm_fuse_pass");
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fused_nodes_after = GetNumOpNodes(graph, "skip_layernorm");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 4,
                    platform::errors::PreconditionNotMet(
                        "The number of nodes before and after the fuse does "
                        "not meet expectations"));
  PADDLE_ENFORCE_EQ(
      num_fused_nodes_after, 1,
      platform::errors::PreconditionNotMet(
          "The number of fusion nodes does not meet expectations after fuse"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(skip_layernorm_fuse_pass);
