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

#include "paddle/fluid/framework/ir/embedding_eltwise_layernorm_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {

TEST(EmbeddingElewiseLayernormFusePass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x, y)                       elementwise_add    -> elementwise_out
  // (elementwise_out, scale, bias) layer_norm       -> layer_norm_out...
  Layers layers;
  auto* x0 = layers.data("x0", {1, 256, 1});
  auto* x1 = layers.data("x1", {1, 256, 1});
  auto* x2 = layers.data("x2", {1, 256, 1});
  auto* x3 = layers.data("x3", {1, 256, 1});

  auto* emb0 = layers.data("emb0", {18000, 768}, true);
  auto* emb1 = layers.data("emb1", {4, 768}, true);
  auto* emb2 = layers.data("emb2", {513, 768}, true);
  auto* emb3 = layers.data("emb3", {3, 768}, true);

  auto* lkt0 = layers.embedding(x0, emb0);
  auto* lkt1 = layers.embedding(x1, emb1);
  auto* lkt2 = layers.embedding(x2, emb2);
  auto* lkt3 = layers.embedding(x3, emb3);

  auto* elementwise_out1 = layers.elementwise_add(lkt0, lkt2);
  auto* elementwise_out2 = layers.elementwise_add(elementwise_out1, lkt1);
  auto* elementwise_out3 = layers.elementwise_add(elementwise_out2, lkt3);

  auto* scale = layers.data("scale", {768}, true);
  auto* bias = layers.data("bias", {768}, true);
  layers.layer_norm(elementwise_out3, scale, bias);

  auto* y0 = layers.data("y0", {1, 256, 1});
  auto* y1 = layers.data("y1", {1, 256, 1});
  auto* y2 = layers.data("y2", {1, 256, 1});

  auto* emb0y = layers.data("emb0y", {18000, 768}, true);
  auto* emb1y = layers.data("emb1y", {4, 768}, true);
  auto* emb2y = layers.data("emb2y", {513, 768}, true);

  auto* lkt0y = layers.embedding(y0, emb0y);
  auto* lkt1y = layers.embedding(y1, emb1y);
  auto* lkt2y = layers.embedding(y2, emb2y);

  auto* elementwise_out1y = layers.elementwise_add(lkt0y, lkt2y);
  auto* elementwise_out2y = layers.elementwise_add(elementwise_out1y, lkt1y);

  auto* scaley = layers.data("scaley", {768}, true);
  auto* biasy = layers.data("biasy", {768}, true);
  layers.layer_norm(elementwise_out2y, scaley, biasy);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass =
      PassRegistry::Instance().Get("embedding_eltwise_layernorm_fuse_pass");
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fused_nodes_after =
      GetNumOpNodes(graph, "fused_embedding_eltwise_layernorm");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after + 28,
                    common::errors::PreconditionNotMet(
                        "The number of nodes before and after the fuse does "
                        "not meet expectations"));
  PADDLE_ENFORCE_EQ(
      num_fused_nodes_after,
      2,
      common::errors::PreconditionNotMet(
          "The number of fusion nodes does not meet expectations after fuse"));
}

TEST(EmbeddingElewiseLayernormFusePass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("embedding_eltwise_layernorm_fuse_pass"));
}

}  // namespace paddle::framework::ir

USE_PASS(embedding_eltwise_layernorm_fuse_pass);
