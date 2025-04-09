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

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/ir/repeated_fc_relu_fuse_pass.h"

namespace paddle::framework {
class VarDesc;
}  // namespace paddle::framework

namespace paddle::framework::ir {

void TestMain(int num_fc) {
  // inputs                                 operator    output
  // -------------------------------------------------------------
  // (x, filters, bias_0)                   conv2d   -> conv2d_out
  // (conv2d_out, fc_weights_0, fc_bias_0)  fc       -> fc_out_0
  // (fc_out_0, fc_weights_1, fc_bias_1)    fc       -> fc_out_1
  // ...
  Layers layers;
  VarDesc* x = layers.data("x");
  VarDesc* filters = layers.data("filters", {}, true);
  VarDesc* bias_0 = layers.data("bias_0", {}, true);
  VarDesc* conv2d_out = layers.conv2d(x, filters, bias_0);
  VarDesc* fc_in = conv2d_out;
  for (int i = 0; i < num_fc; ++i) {
    VarDesc* weights_i =
        layers.data("fc_weights_" + std::to_string(i), {}, true);
    VarDesc* bias_i = layers.data("fc_bias_" + std::to_string(i), {}, true);
    std::string activation_type = i < (num_fc - 1) ? "relu" : "";
    VarDesc* fc_out = layers.fc(fc_in, weights_i, bias_i, 1, activation_type);
    fc_in = fc_out;
  }

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("repeated_fc_relu_fuse_pass");
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  int num_fc_nodes_before = GetNumOpNodes(graph, "fc");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_fused_nodes_after = GetNumOpNodes(graph, "fusion_repeated_fc_relu");
  VLOG(3) << DebugString(graph);

  // Delete (num_fc_nodes_before - 1) fc ops
  PADDLE_ENFORCE_EQ(
      num_nodes_before - (num_fc_nodes_before - 1) + 1,
      num_nodes_after,
      common::errors::InvalidArgument(
          "num_nodes_before = %d, num_fc_nodes_before = %d, num_nodes_after = "
          "%d.",
          num_nodes_before,
          num_fc_nodes_before,
          num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fused_nodes_after,
                    1,
                    common::errors::InvalidArgument(
                        "num_fused_nodes_after = %d.", num_fused_nodes_after));
}

TEST(RepeatedFCReluFusePass, basic_3) { TestMain(3); }

TEST(RepeatedFCReluFusePass, basic_9) { TestMain(9); }

}  // namespace paddle::framework::ir

USE_PASS(repeated_fc_relu_fuse_pass);
