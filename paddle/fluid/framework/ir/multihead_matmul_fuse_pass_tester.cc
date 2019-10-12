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

#include "paddle/fluid/framework/ir/multihead_matmul_fuse_pass.h"  // NOLINT
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(MultiHeadMatmulFusePass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x)           layer_norm               -> fc_out_0
  // (fc_out_0, weights_1, bias_1)    fc               -> fc_out_1
  // (fc_out_1, y)                    elementwise_add  -> elementwise_out
  // (elementwise_out, scale, bias_2) layer_norm       ->
  Layers layers;
  auto* x = layers.data("x", {128, 768});
  auto out = layers.layer_norm(x);
  auto* layer_out = out[0];

  auto* weights_0 = layers.data("weights0", {768, 768}, true);
  auto* weights_1 = layers.data("weights1", {768, 768}, true);
  auto* weights_2 = layers.data("weights2", {768, 768}, true);

  auto* mul_out_0 = layers.mul(layer_out, weights_0);
  auto* mul_out_1 = layers.mul(layer_out, weights_1);
  auto* mul_out_2 = layers.mul(layer_out, weights_2);

  auto* b0 = layers.data("bias_0", {768}, true);
  auto* b1 = layers.data("bias_1", {768}, true);
  auto* b2 = layers.data("bias_2", {768}, true);

  auto* elementwise_out_0 = layers.elementwise_add(mul_out_0, b0);
  auto* elementwise_out_1 = layers.elementwise_add(mul_out_1, b1);
  auto* elementwise_out_2 = layers.elementwise_add(mul_out_2, b2);

  std::vector<int> shape = {128, 12, 64};
  auto* reshape_0 = layers.reshape2(elementwise_out_0, shape);
  auto* reshape_1 = layers.reshape2(elementwise_out_1, shape);
  auto* reshape_2 = layers.reshape2(elementwise_out_2, shape);

  std::vector<int> axis = {0, 2, 1, 3};
  auto* transpose_0 = layers.transpose2(reshape_0, axis);
  auto* transpose_1 = layers.transpose2(reshape_1, axis);
  auto* transpose_2 = layers.transpose2(reshape_2, axis);

  auto* scale_0 = layers.scale(transpose_0, 0.125, 0, false);
  auto* matmul_qk = layers.matmul(scale_0, transpose_1);

  auto* bqk = layers.data("biasqk", {768}, true);
  auto* elementwise_qk = layers.elementwise_add(matmul_qk, bqk);
  auto* softmax_qk = layers.softmax(elementwise_qk, -1);

  auto* matmul_qkv = layers.matmul(softmax_qk, transpose_2);

  auto* transpose_qkv = layers.transpose2(matmul_qkv, {0, 2, 1, 3});
  auto* reshape_qkv_out = layers.reshape2(transpose_qkv, {128, 768});
  auto* weights_l = layers.data("weightsl", {768, 768}, true);
  layers.mul(reshape_qkv_out, weights_l);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("multihead_matmul_fuse_pass");
  int num_nodes_before = graph->Nodes().size();
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fused_nodes_after = GetNumOpNodes(graph, "multihead_matmul");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 29);
  PADDLE_ENFORCE_EQ(num_fused_nodes_after, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(multihead_matmul_fuse_pass);
