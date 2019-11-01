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

#include "paddle/fluid/framework/ir/fusion_group/fusion_group_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(FusionGroupPass, elementwise_list) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (x, y)                     mul              -> tmp_0
  // (tmp_0, z)                 elementwise_add  -> tmp_1
  // tmp_1                      relu             -> tmp_2
  // (tmp_2, w)                 elementwise_add  -> tmp_3
  //
  // Expression: tmp_3 = relu(mul(x, y) + z) + w
  Layers layers;
  auto* x = layers.data("x", {16, 16});
  auto* y = layers.data("y", {16, 32});
  auto* tmp_0 = layers.mul(x, y);
  tmp_0->SetShape({16, 32});
  auto* z = layers.data("z", {16, 32});
  auto* tmp_1 = layers.elementwise_add(tmp_0, z);
  auto* tmp_2 = layers.relu(tmp_1);
  tmp_2->SetShape({16, 32});
  auto* w = layers.data("w", {16, 32});
  layers.elementwise_add(tmp_2, w);

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));

  // The following codes is to insert a graph_viz_pass to transform the graph to
  // a .dot file. It is used for debug.
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path", new
  // std::string("00_elementwise_list.dot"));
  // graph.reset(graph_viz_pass->Apply(graph.release()));

  auto fusion_group_pass = PassRegistry::Instance().Get("fusion_group_pass");
  VLOG(3) << DebugString(graph);

  graph.reset(fusion_group_pass->Apply(graph.release()));
  int num_fusion_group_ops = GetNumOpNodes(graph, "fusion_group");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_fusion_group_ops, 1);

  // The following codes is to insert a graph_viz_pass to transform the graph to
  // a .dot file. It is used for debug.
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path", new
  // std::string("01_elementwise_list.fusion_group.dot"));
  // graph.reset(graph_viz_pass->Apply(graph.release()));
}

TEST(FusionGroupPass, elementwise_tree) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (x0, y0)                   mul              -> tmp_0
  // x1                         sigmoid          -> tmp_1
  // (tmp_0, tmp_1)             elementwise_mul  -> tmp_2
  // x2                         sigmoid          -> tmp_3
  // x3                         tanh             -> tmp_4
  // (tmp_3, tmp_4)             elementwise_mul  -> tmp_5
  // (tmp_2, tmp_5)             elementwise_add  -> tmp_6
  // x4                         tanh             -> tmp_7
  // x5                         sigmoid          -> tmp_8
  // (tmp_7, tmp_8)             elementwise_mul  -> tmp_9
  // (tmp_6, tmp_9)             mul              -> tmp_10
  //
  // Expression: tmp_6 = mul(x0, y0) * sigmoid(x1) + sigmoid(x2) * tanh(x3)
  //             tmp_9 = tanh(x4) * sigmoid(x5)
  //             tmp_10 = mul(tmp_6, tmp_9)
  Layers layers;
  auto* x0 = layers.data("x0", {16, 16});
  auto* y0 = layers.data("y0", {16, 32});
  auto* tmp_0 = layers.mul(x0, y0);
  tmp_0->SetShape({16, 32});

  auto* x1 = layers.data("x1", {16, 32});
  auto* tmp_1 = layers.sigmoid(x1);
  tmp_1->SetShape({16, 32});

  auto* tmp_2 = layers.elementwise_mul(tmp_0, tmp_1);
  tmp_2->SetShape({16, 32});

  auto* x2 = layers.data("x2", {16, 32});
  auto* tmp_3 = layers.sigmoid(x2);
  tmp_3->SetShape({16, 32});
  auto* x3 = layers.data("x3", {16, 32});
  auto* tmp_4 = layers.tanh(x3);
  tmp_4->SetShape({16, 32});
  auto* tmp_5 = layers.elementwise_mul(tmp_3, tmp_4);
  tmp_5->SetShape({16, 32});

  auto* tmp_6 = layers.elementwise_add(tmp_2, tmp_5);
  tmp_6->SetShape({16, 32});

  auto* x4 = layers.data("x4", {16, 32});
  auto* tmp_7 = layers.tanh(x4);
  tmp_7->SetShape({16, 32});
  auto* x5 = layers.data("x5", {16, 32});
  auto* tmp_8 = layers.sigmoid(x5);
  tmp_8->SetShape({16, 32});

  auto* tmp_9 = layers.elementwise_mul(tmp_7, tmp_8);
  tmp_9->SetShape({16, 32});
  layers.mul(tmp_6, tmp_9);

  std::unique_ptr<Graph> graph(new Graph(layers.main_program()));

  // The following codes is to insert a graph_viz_pass to transform the graph to
  // a .dot file. It is used for debug.
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path", new
  // std::string("00_elementwise_tree.dot"));
  // graph.reset(graph_viz_pass->Apply(graph.release()));

  auto fusion_group_pass = PassRegistry::Instance().Get("fusion_group_pass");
  LOG(INFO) << DebugString(graph);

  graph.reset(fusion_group_pass->Apply(graph.release()));
  int num_fusion_group_ops = GetNumOpNodes(graph, "fusion_group");
  LOG(INFO) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_fusion_group_ops, 2);

  // The following codes is to insert a graph_viz_pass to transform the graph to
  // a .dot file. It is used for debug.
  // auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  // graph_viz_pass->Set("graph_viz_path", new
  // std::string("01_elementwise_tree.fusion_group.dot"));
  // graph.reset(graph_viz_pass->Apply(graph.release()));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fusion_group_pass);
USE_PASS(graph_viz_pass);
