// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/xpu/multi_encoder_xpu_adaptive_seqlen_fuse_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct MultiEncoderXPUAdaptiveSeqlenPattern : public PatternBase {
  MultiEncoderXPUAdaptiveSeqlenPattern(PDPattern* pattern,
                                       const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(embedding_xpu);
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(matmul);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(stack);
  PATTERN_DECL_NODE(multi_encoder_xpu);
  // declare variable node's name
  PATTERN_DECL_NODE(mask);
  PATTERN_DECL_NODE(embedding_xpu_out);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(matmul_out);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(stack_out);
};

MultiEncoderXPUAdaptiveSeqlenPattern::MultiEncoderXPUAdaptiveSeqlenPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* embedding_xpu = pattern->NewNode(embedding_xpu_repr())
                            ->assert_is_op("embedding_with_eltwise_add_xpu");
  auto* embedding_xpu_out =
      pattern->NewNode(embedding_xpu_out_repr())
          ->assert_is_op_output("embedding_with_eltwise_add_xpu", "out")
          ->assert_is_op_input("layer_norm", "X");
  auto* layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto* layer_norm_out = pattern->NewNode(layer_norm_out_repr())
                             ->assert_is_op_output("layer_norm", "Y")
                             ->assert_is_op_input("multi_encoder_xpu", "x");

  auto* mask = pattern->NewNode(mask_repr())
                   ->assert_is_op_input("matmul", "X")
                   ->assert_is_op_input("matmul", "Y");
  auto* matmul = pattern->NewNode(matmul_repr())->assert_is_op("matmul");
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output("matmul", "Out")
                         ->assert_is_op_input("scale", "X");
  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out = pattern->NewNode(scale_out_repr())
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input("stack", "X");
  auto* stack = pattern->NewNode(stack_repr())->assert_is_op("stack");
  auto* stack_out = pattern->NewNode(stack_out_repr())
                        ->assert_is_op_output("stack", "Y")
                        ->assert_is_op_input("multi_encoder_xpu", "mask");

  auto* multi_encoder_xpu = pattern->NewNode(multi_encoder_xpu_repr())
                                ->assert_is_op("multi_encoder_xpu");

  embedding_xpu->LinksTo({embedding_xpu_out});
  layer_norm->LinksFrom({embedding_xpu_out}).LinksTo({layer_norm_out});
  matmul->LinksFrom({mask}).LinksTo({matmul_out});
  scale->LinksFrom({matmul_out}).LinksTo({scale_out});
  stack->LinksFrom({scale_out}).LinksTo({stack_out});
  multi_encoder_xpu->LinksFrom({layer_norm_out, stack_out});
}

}  // namespace patterns

int MultiEncoderXPUAdaptiveSeqlenFusePass::ApplyAdaptiveSeqlenPassV1(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::MultiEncoderXPUAdaptiveSeqlenPattern pattern(gpd.mutable_pattern(),
                                                         name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle MultiEncoderXPUAdaptiveSeqlenFusePass fuse";
    GET_IR_NODE(embedding_xpu);
    GET_IR_NODE(layer_norm);
    GET_IR_NODE(matmul);
    GET_IR_NODE(scale);
    GET_IR_NODE(stack);
    GET_IR_NODE(multi_encoder_xpu);
    GET_IR_NODE(mask);
    GET_IR_NODE(embedding_xpu_out);
    GET_IR_NODE(layer_norm_out);
    GET_IR_NODE(matmul_out);
    GET_IR_NODE(scale_out);
    GET_IR_NODE(stack_out);

    std::string mask_name = mask->Name();
    std::string seq_lod_name = mask_name + "_seq_lod";
    VarDesc seq_lod_desc(seq_lod_name);
    auto* seq_lod = graph->CreateVarNode(&seq_lod_desc);
    std::string max_seq_len_name = mask_name + "_max_seq_len";
    VarDesc max_seq_len_desc(max_seq_len_name);
    auto* max_seq_len = graph->CreateVarNode(&max_seq_len_desc);

    embedding_xpu->Op()->SetInput("mask", {mask_name});
    embedding_xpu->Op()->SetOutput("seq_lod", {seq_lod_name});
    embedding_xpu->Op()->SetOutput("max_seq_len", {max_seq_len_name});
    multi_encoder_xpu->Op()->SetInput("seq_lod", {seq_lod_name});
    multi_encoder_xpu->Op()->SetInput("max_seq_len", {max_seq_len_name});
    multi_encoder_xpu->Op()->RemoveInput("mask");
    IR_NODE_LINK_TO(mask, embedding_xpu);
    IR_NODE_LINK_TO(embedding_xpu, seq_lod);
    IR_NODE_LINK_TO(embedding_xpu, max_seq_len);
    IR_NODE_LINK_TO(seq_lod, multi_encoder_xpu);
    IR_NODE_LINK_TO(max_seq_len, multi_encoder_xpu);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes{
        matmul, scale, stack, matmul_out, scale_out, stack_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void MultiEncoderXPUAdaptiveSeqlenFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = ApplyAdaptiveSeqlenPassV1(graph);

  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_encoder_xpu_adaptive_seqlen_fuse_pass,
              paddle::framework::ir::MultiEncoderXPUAdaptiveSeqlenFusePass);

REGISTER_PASS_CAPABILITY(multi_encoder_xpu_adaptive_seqlen_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "multi_encoder_xpu", 0));
