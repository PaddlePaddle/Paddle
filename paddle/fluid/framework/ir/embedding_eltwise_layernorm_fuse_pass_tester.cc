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

#include "paddle/fluid/framework/ir/embedding_eltwise_layernorm_fuse_pass.h"  // NOLINT
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(EmbeddingEltwiseLayernormFusePass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (word, weights_0)                lookup_table     ->  word_emb
  // (pos, weights_1)                 lookup_table     ->  pos_emb
  // (sent, weights_2)                lookup_table     ->  sent_emb
  // (word_emb, pos_emb)              elementweise_add -> elementwise_out_0
  // (elemtwise_out_0, sent_emb)      elementweise_add -> elementwise_out_1
  // (elementwise_out_1)              layer_norm       -> layer_norm_out
  Layers layers;
  auto* word = layers.data("word", {1, 128, 1});
  auto* pos = layers.data("pos", {1, 128, 1});
  auto* sent = layers.data("sent", {1, 128, 1});

  auto* weights_0 = layers.data("weights0", {128, 768}, true);
  auto* weights_1 = layers.data("weights1", {128, 768}, true);
  auto* weights_2 = layers.data("weights2", {128, 768}, true);
  auto* scale = layers.data("scale", {768}, true);
  auto* bias = layers.data("bias", {768}, true);

  auto* word_emb = layers.embedding(word, weights_0);
  auto* pos_emb = layers.embedding(pos, weights_1);
  auto* sent_emb = layers.embedding(sent, weights_2);

  auto* elementwise_out_0 = layers.elementwise_add(word_emb, pos_emb);
  auto* elementwise_out_1 = layers.elementwise_add(elementwise_out_0, sent_emb);

  layers.layer_norm(elementwise_out_1, scale, bias);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));

  auto pass =
      PassRegistry::Instance().Get("embedding_eltwise_layernorm_fuse_pass");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_fused_nodes_after =
      GetNumOpNodes(graph, "fused_embedding_eltwise_layernorm");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(
      num_fused_nodes_after, 1,
      platform::errors::InvalidArgument(
          "After the embedding_eltwise_layernorm pass, there should "
          "be one embedding_eltwise_layernorm op, but result is  %d",
          num_fused_nodes_after));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(embedding_eltwise_layernorm_fuse_pass);
