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

#include "paddle/fluid/framework/ir/swin_attention_biasqk_fold_pass.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#include <string>

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                   \
  GET_IR_NODE(elementwise_00_op);   \
  GET_IR_NODE(elementwise_00_in_y); \
  GET_IR_NODE(elementwise_00_out);  \
  GET_IR_NODE(unsqueeze_01_op);     \
  GET_IR_NODE(unsqueeze_01_op_x);   \
  GET_IR_NODE(unsqueeze_01_out);    \
  GET_IR_NODE(reshape_10_op);       \
  GET_IR_NODE(reshape_10_out);      \
  GET_IR_NODE(unsqueeze_11_op);     \
  GET_IR_NODE(unsqueeze_11_out);    \
  GET_IR_NODE(elementwise_20_op);   \
  GET_IR_NODE(elementwise_20_out);  \
  GET_IR_NODE(reshape_30_op);       \
  GET_IR_NODE(reshape_30_out);

namespace paddle {
namespace framework {
namespace ir {
void SwinAttentionBiasqkFoldPass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  const std::string pattern_name = "swin_attention_bisqk_fold";
  FusePassBase::Init(pattern_name, graph);
  // auto* scope = param_scope();

  // std::unordered_set<std::string> matmul_ops{"matmul", "matmul_v2"};
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_op_input("elementwise_add", "X")
                  ->AsInput();
  patterns::SwinAttentionBiasQkFold pattern(gpd.mutable_pattern(),
                                            pattern_name);
  pattern(x);
  int fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;
    // do nothing for test
    // OpDesc desc(elementwise_00_op->Op()->Block());
    // desc.SetType("elementwise_add");
    // desc.SetInput("X",{subgraph.at(x)->Name()});
    // desc.SetAttr("BiasQK_swinFolded",true);
    auto* elementwise_00_op_desc = elementwise_00_op->Op();

    elementwise_00_op_desc->SetInput("BiasQK_mask",{unsqueeze_01_op_x->Name()});
    elementwise_00_op_desc->SetOutput("Out", {reshape_30_out->Name()});
    IR_NODE_LINK_TO(unsqueeze_01_op_x,elementwise_00_op);
    IR_NODE_LINK_TO(elementwise_00_op, reshape_30_out);

    std::unordered_set<const Node*> marked_nodes({// unsqueeze_01_op_x,
                                                  // elementwise_00_op,
                                                  elementwise_00_out,
                                                  unsqueeze_01_op,
                                                  unsqueeze_01_out,
                                                  reshape_10_op,
                                                  reshape_10_out,
                                                  unsqueeze_11_op,
                                                  unsqueeze_11_out,
                                                  elementwise_20_op,
                                                  elementwise_20_out,
                                                  reshape_30_op});

    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);
  AddStatis(fusion_count);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(swin_attention_biasqk_fold_pass,
              paddle::framework::ir::SwinAttentionBiasqkFoldPass);

REGISTER_PASS_CAPABILITY(swin_attention_biasqk_fold_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().GE(
            "reshape2", 0));
