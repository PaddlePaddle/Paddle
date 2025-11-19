/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/dense_multihead_matmul_to_sparse_pass.h"  // NOLINT
#include "paddle/fluid/framework/ir/multihead_matmul_fuse_pass.h"  // NOLINT
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_version_registry.h"

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
  AddVarToScope(param_scope, "weights0_sparse_2_4", {768, 768});
  AddVarToScope(param_scope, "weights1_sparse_2_4", {768, 768});
  AddVarToScope(param_scope, "weights2_sparse_2_4", {768, 768});

  AddVarToScope(param_scope, "bias_0", {768});
  AddVarToScope(param_scope, "bias_1", {768});
  AddVarToScope(param_scope, "bias_2", {768});
  AddVarToScope(param_scope, "biasqk", {768});
  AddVarToScope(param_scope, "weightsl", {768, 768});
  return param_scope;
}

TEST(DenseMultiHeadMatmulToSparsePass, basic) {
  // inputs                           operator            output
  // --------------------------------------------------------------------
  // (x)                              layer_norm       -> layer_norm_out
  // (layer_norm_out, weights_0_sparse_2_4)      mul              -> mul_out0
  // (layer_norm_out, weights_1_sparse_2_4)      mul              -> mul_out1
  // (layer_norm_out, weights_2_sparse_2_4)      mul              -> mul_out2
  // (mul_out0, bias_0)               elementweise_add -> eltadd_0
  // (mul_out1, bias_1)               elementweise_add -> eltadd_1
  // (mul_out2, bias_2)               elementweise_add -> eltadd_2
  // (eltadd_0)                       reshape2         -> reshape_0
  // (eltadd_1)                       reshape2         -> reshape_1
  // (eltadd_2)                       reshape2         -> reshape_2
  // (reshape_0)                      transpose2       -> transpose_0
  // (reshape_1)                      transpose2       -> transpose_1
  // (reshape_2)                      transpose2       -> transpose_2
  // (transpose_0)                    scale            -> scale_0
  // (scale_0, transpose_1)           matmul           -> matmul_qk
  // (matmul_qk, bias_qk)             elementwise_add  -> eltadd_qk
  // (eltadd_qk)                      softmax          -> softmax_qk
  // (softmax_qk, transpose_2)        matmul           -> matmul_qkv
  // (matmul_qkv)                     transpose        -> transpose_qkv
  // (transpose_qkv)                  reshape          -> reshape_qkv
  // (reshape_qkv)                    mul              -> mul_qkv
  Layers layers;
  auto* x = layers.data("x", {1, 128, 768});
  auto out = layers.layer_norm(x);
  auto* layer_out = out[0];

  auto* weights_0 = layers.data("weights0_sparse_2_4", {768, 768}, true);
  auto* weights_1 = layers.data("weights1_sparse_2_4", {768, 768}, true);
  auto* weights_2 = layers.data("weights2_sparse_2_4", {768, 768}, true);

  auto* mul_out_0 = layers.mul(layer_out, weights_0, nullptr, 2);
  auto* mul_out_1 = layers.mul(layer_out, weights_1, nullptr, 2);
  auto* mul_out_2 = layers.mul(layer_out, weights_2, nullptr, 2);

  auto* b0 = layers.data("bias_0", {768}, true);
  auto* b1 = layers.data("bias_1", {768}, true);
  auto* b2 = layers.data("bias_2", {768}, true);

  auto* elementwise_out_0 = layers.elementwise_add(mul_out_0, b0, nullptr, 2);
  auto* elementwise_out_1 = layers.elementwise_add(mul_out_1, b1, nullptr, 2);
  auto* elementwise_out_2 = layers.elementwise_add(mul_out_2, b2, nullptr, 2);

  std::vector<int> shape = {1, 128, 12, 64};
  auto* reshape_0 = layers.reshape2(elementwise_out_0, shape, true);
  auto* reshape_1 = layers.reshape2(elementwise_out_1, shape, true);
  auto* reshape_2 = layers.reshape2(elementwise_out_2, shape, true);

  std::vector<int> axis = {0, 2, 1, 3};
  auto* transpose_0 = layers.transpose2(reshape_0, axis, true);
  auto* transpose_1 = layers.transpose2(reshape_1, axis, true);
  auto* transpose_2 = layers.transpose2(reshape_2, axis, true);

  auto* scale_0 = layers.scale(transpose_0, 0.125, 0, false);
  auto* matmul_qk = layers.matmul(scale_0, transpose_1, nullptr, false, true);

  auto* bqk = layers.data("biasqk", {1, 12, 128, 128}, true);
  auto* elementwise_qk = layers.elementwise_add(matmul_qk, bqk);
  auto* softmax_qk = layers.softmax(elementwise_qk, -1);

  auto* matmul_qkv = layers.matmul(softmax_qk, transpose_2);

  auto* transpose_qkv = layers.transpose2(matmul_qkv, {0, 2, 1, 3}, true);
  auto* reshape_qkv_out = layers.reshape2(transpose_qkv, {1, 128, 768}, true);
  auto* weights_l = layers.data("weightsl", {768, 768}, true);
  layers.mul(reshape_qkv_out, weights_l, nullptr, 2);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  graph->Set("__param_scope__", CreateParamScope());

  auto fuse_pass =
      PassRegistry::Instance().Get("multihead_matmul_fuse_pass_v2");
  auto sparse_pass =
      PassRegistry::Instance().Get("dense_multihead_matmul_to_sparse_pass");

  if (fuse_pass.get() == nullptr || sparse_pass.get() == nullptr)
    LOG(INFO) << "asdfasdf";
  int num_nodes_before = static_cast<int>(graph->Nodes().size());
  VLOG(3) << DebugString(graph);

  graph.reset(fuse_pass->Apply(graph.release()));
  graph.reset(sparse_pass->Apply(graph.release()));
  int num_nodes_after = static_cast<int>(graph->Nodes().size());
  int num_fused_nodes_after = GetNumOpNodes(graph, "sparse_multihead_matmul");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before,
                    num_nodes_after + 39,
                    common::errors::InvalidArgument(
                        "After the multihead_matmul pass and sparse pass, The "
                        "node num in graph "
                        "should be %d, but the result is %d",
                        num_nodes_before - 39,
                        num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fused_nodes_after,
                    1,
                    common::errors::InvalidArgument(
                        "After the multihead_matmul pass and sparse pass, "
                        "there should be one "
                        "sparse_multihead_matmul op, but the result is %d",
                        num_fused_nodes_after));
}

}  // namespace paddle::framework::ir

USE_PASS(multihead_matmul_fuse_pass);
USE_PASS(multihead_matmul_fuse_pass_v2);
USE_PASS(dense_multihead_matmul_to_sparse_pass);
