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

struct AdaptiveSeqlenPatternV1 : public PatternBase {
  AdaptiveSeqlenPatternV1(PDPattern* pattern,
                          const std::string& name_scope,
                          const std::string& matmul_type);

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

AdaptiveSeqlenPatternV1::AdaptiveSeqlenPatternV1(PDPattern* pattern,
                                                 const std::string& name_scope,
                                                 const std::string& matmul_type)
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
                   ->assert_is_op_input(matmul_type, "X")
                   ->assert_is_op_input(matmul_type, "Y");
  auto* matmul = pattern->NewNode(matmul_repr())->assert_is_op(matmul_type);
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output(matmul_type, "Out")
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
    ir::Graph* graph, const std::string& matmul_type) const {
  GraphPatternDetector gpd;
  patterns::AdaptiveSeqlenPatternV1 pattern(
      gpd.mutable_pattern(), name_scope_, matmul_type);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyAdaptiveSeqlenPassV1 fuse";
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

namespace patterns {

struct AdaptiveSeqlenPatternV2 : public PatternBase {
  AdaptiveSeqlenPatternV2(PDPattern* pattern,
                          const std::string& name_scope,
                          const std::string& matmul_type);

  // declare operator node's name
  PATTERN_DECL_NODE(embedding_xpu);
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(not_equal);
  PATTERN_DECL_NODE(cast);
  PATTERN_DECL_NODE(unsqueeze_0);
  PATTERN_DECL_NODE(matmul);
  PATTERN_DECL_NODE(scale_0);
  PATTERN_DECL_NODE(scale_1);
  PATTERN_DECL_NODE(unsqueeze_1);
  PATTERN_DECL_NODE(tile);
  PATTERN_DECL_NODE(multi_encoder_xpu);
  // declare variable node's name
  PATTERN_DECL_NODE(mask);
  PATTERN_DECL_NODE(embedding_xpu_out);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(not_equal_out);
  PATTERN_DECL_NODE(cast_out);
  PATTERN_DECL_NODE(unsqueeze_0_out);
  PATTERN_DECL_NODE(matmul_out);
  PATTERN_DECL_NODE(scale_0_out);
  PATTERN_DECL_NODE(scale_1_out);
  PATTERN_DECL_NODE(unsqueeze_1_out);
  PATTERN_DECL_NODE(tile_out);
};

AdaptiveSeqlenPatternV2::AdaptiveSeqlenPatternV2(PDPattern* pattern,
                                                 const std::string& name_scope,
                                                 const std::string& matmul_type)
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

  auto* mask =
      pattern->NewNode(mask_repr())->assert_is_op_input("not_equal", "X");
  auto* not_equal =
      pattern->NewNode(not_equal_repr())->assert_is_op("not_equal");
  auto* not_equal_out = pattern->NewNode(not_equal_out_repr())
                            ->assert_is_op_output("not_equal", "Out")
                            ->assert_is_op_input("cast", "X");
  auto* cast = pattern->NewNode(cast_repr())->assert_is_op("cast");
  auto* cast_out = pattern->NewNode(cast_out_repr())
                       ->assert_is_op_output("cast", "Out")
                       ->assert_is_op_input("unsqueeze2", "X");
  auto* unsqueeze_0 =
      pattern->NewNode(unsqueeze_0_repr())->assert_is_op("unsqueeze2");
  auto* unsqueeze_0_out = pattern->NewNode(unsqueeze_0_out_repr())
                              ->assert_is_op_output("unsqueeze2", "Out")
                              ->assert_is_op_input(matmul_type, "X")
                              ->assert_is_op_input(matmul_type, "Y");
  auto* matmul = pattern->NewNode(matmul_repr())->assert_is_op(matmul_type);
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output(matmul_type, "Out")
                         ->assert_is_op_input("scale", "X");
  auto* scale_0 = pattern->NewNode(scale_0_repr())->assert_is_op("scale");
  auto* scale_0_out = pattern->NewNode(scale_0_out_repr())
                          ->assert_is_op_output("scale", "Out")
                          ->assert_is_op_input("scale", "X");
  auto* scale_1 = pattern->NewNode(scale_1_repr())->assert_is_op("scale");
  auto* scale_1_out = pattern->NewNode(scale_1_out_repr())
                          ->assert_is_op_output("scale", "Out")
                          ->assert_is_op_input("unsqueeze2", "X");
  auto* unsqueeze_1 =
      pattern->NewNode(unsqueeze_1_repr())->assert_is_op("unsqueeze2");
  auto* unsqueeze_1_out = pattern->NewNode(unsqueeze_1_out_repr())
                              ->assert_is_op_output("unsqueeze2", "Out")
                              ->assert_is_op_input("tile", "X");
  auto* tile = pattern->NewNode(tile_repr())->assert_is_op("tile");
  auto* tile_out = pattern->NewNode(tile_out_repr())
                       ->assert_is_op_output("tile", "Out")
                       ->assert_is_op_input("multi_encoder_xpu", "mask");

  auto* multi_encoder_xpu = pattern->NewNode(multi_encoder_xpu_repr())
                                ->assert_is_op("multi_encoder_xpu");

  embedding_xpu->LinksTo({embedding_xpu_out});
  layer_norm->LinksFrom({embedding_xpu_out}).LinksTo({layer_norm_out});
  not_equal->LinksFrom({mask}).LinksTo({not_equal_out});
  cast->LinksFrom({not_equal_out}).LinksTo({cast_out});
  unsqueeze_0->LinksFrom({cast_out}).LinksTo({unsqueeze_0_out});
  matmul->LinksFrom({unsqueeze_0_out}).LinksTo({matmul_out});
  scale_0->LinksFrom({matmul_out}).LinksTo({scale_0_out});
  scale_1->LinksFrom({scale_0_out}).LinksTo({scale_1_out});
  unsqueeze_1->LinksFrom({scale_1_out}).LinksTo({unsqueeze_1_out});
  tile->LinksFrom({unsqueeze_1_out}).LinksTo({tile_out});
  multi_encoder_xpu->LinksFrom({layer_norm_out, tile_out});
}

}  // namespace patterns

int MultiEncoderXPUAdaptiveSeqlenFusePass::ApplyAdaptiveSeqlenPassV2(
    ir::Graph* graph, const std::string& matmul_type) const {
  GraphPatternDetector gpd;
  patterns::AdaptiveSeqlenPatternV2 pattern(
      gpd.mutable_pattern(), name_scope_, matmul_type);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyAdaptiveSeqlenPassV2 fuse";
    GET_IR_NODE(embedding_xpu);
    GET_IR_NODE(layer_norm);
    GET_IR_NODE(not_equal);
    GET_IR_NODE(cast);
    GET_IR_NODE(unsqueeze_0);
    GET_IR_NODE(matmul);
    GET_IR_NODE(scale_0);
    GET_IR_NODE(scale_1);
    GET_IR_NODE(unsqueeze_1);
    GET_IR_NODE(tile);
    GET_IR_NODE(multi_encoder_xpu);
    GET_IR_NODE(mask);
    GET_IR_NODE(embedding_xpu_out);
    GET_IR_NODE(layer_norm_out);
    GET_IR_NODE(not_equal_out);
    GET_IR_NODE(cast_out);
    GET_IR_NODE(unsqueeze_0_out);
    GET_IR_NODE(matmul_out);
    GET_IR_NODE(scale_0_out);
    GET_IR_NODE(scale_1_out);
    GET_IR_NODE(unsqueeze_1_out);
    GET_IR_NODE(tile_out);

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
    std::unordered_set<const Node*> delete_nodes{not_equal,
                                                 cast,
                                                 unsqueeze_0,
                                                 matmul,
                                                 scale_0,
                                                 scale_1,
                                                 unsqueeze_1,
                                                 tile,
                                                 not_equal_out,
                                                 cast_out,
                                                 unsqueeze_0_out,
                                                 matmul_out,
                                                 scale_0_out,
                                                 scale_1_out,
                                                 unsqueeze_1_out,
                                                 tile_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

namespace patterns {
struct AdaptiveSeqlenPatternV3 : public PatternBase {
  AdaptiveSeqlenPatternV3(PDPattern* pattern,
                          const std::string& name_scope,
                          const std::string& matmul_type);

  // declare operator node's name
  PATTERN_DECL_NODE(multi_encoder_xpu);
  PATTERN_DECL_NODE(matmul);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(stack);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(mask);
  PATTERN_DECL_NODE(matmul_out);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(stack_out);
};

AdaptiveSeqlenPatternV3::AdaptiveSeqlenPatternV3(PDPattern* pattern,
                                                 const std::string& name_scope,
                                                 const std::string& matmul_type)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x =
      pattern->NewNode(x_repr())->assert_is_op_input("multi_encoder_xpu", "x");

  auto* mask = pattern->NewNode(mask_repr())
                   ->assert_is_op_input(matmul_type, "X")
                   ->assert_is_op_input(matmul_type, "Y");
  auto* matmul = pattern->NewNode(matmul_repr())->assert_is_op(matmul_type);
  auto* matmul_out = pattern->NewNode(matmul_out_repr())
                         ->assert_is_op_output(matmul_type, "Out")
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

  matmul->LinksFrom({mask}).LinksTo({matmul_out});
  scale->LinksFrom({matmul_out}).LinksTo({scale_out});
  stack->LinksFrom({scale_out}).LinksTo({stack_out});
  multi_encoder_xpu->LinksFrom({x, stack_out});
}

}  // namespace patterns

int MultiEncoderXPUAdaptiveSeqlenFusePass::ApplyAdaptiveSeqlenPassV3(
    ir::Graph* graph, const std::string& matmul_type) const {
  GraphPatternDetector gpd;
  patterns::AdaptiveSeqlenPatternV3 pattern(
      gpd.mutable_pattern(), name_scope_, matmul_type);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle ApplyAdaptiveSeqlenPassV3 fuse";
    GET_IR_NODE(multi_encoder_xpu);
    GET_IR_NODE(matmul);
    GET_IR_NODE(scale);
    GET_IR_NODE(stack);
    GET_IR_NODE(x);
    GET_IR_NODE(mask);
    GET_IR_NODE(matmul_out);
    GET_IR_NODE(scale_out);
    GET_IR_NODE(stack_out);

    std::string mask_name = mask->Name();
    std::string seq_len_name = mask_name + "_seq_len";
    VarDesc seq_len_desc(seq_len_name);
    auto* seq_len = graph->CreateVarNode(&seq_len_desc);
    std::string seq_lod_name = mask_name + "_seq_lod";
    VarDesc seq_lod_desc(seq_lod_name);
    auto* seq_lod = graph->CreateVarNode(&seq_lod_desc);
    std::string max_seq_len_name = mask_name + "_max_seq_len";
    VarDesc max_seq_len_desc(max_seq_len_name);
    auto* max_seq_len = graph->CreateVarNode(&max_seq_len_desc);
    std::string x_vsl_name = x->Name() + "_vsl_packed";
    VarDesc x_vsl_desc(x_vsl_name);
    auto* x_vsl = graph->CreateVarNode(&x_vsl_desc);

    framework::OpDesc op_desc;
    op_desc.SetType("mask_adaptive_xpu");
    op_desc.SetInput("mask", {mask_name});
    op_desc.SetOutput("length", {seq_len_name});
    op_desc.SetOutput("seq_lod", {seq_lod_name});
    op_desc.SetOutput("pad_seq_len", {max_seq_len_name});
    auto* mask_adaptive_xpu = graph->CreateOpNode(&op_desc);

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("sequence_unpad_xpu");
    new_op_desc.SetInput("x", {x->Name()});
    new_op_desc.SetInput("length", {seq_len_name});
    new_op_desc.SetOutput("out", {x_vsl_name});
    auto* sequence_unpad = graph->CreateOpNode(&new_op_desc);

    multi_encoder_xpu->Op()->SetInput("x", {x_vsl_name});
    multi_encoder_xpu->Op()->SetInput("seq_lod", {seq_lod_name});
    multi_encoder_xpu->Op()->SetInput("max_seq_len", {max_seq_len_name});
    multi_encoder_xpu->Op()->RemoveInput("mask");
    IR_NODE_LINK_TO(mask, mask_adaptive_xpu);
    IR_NODE_LINK_TO(mask_adaptive_xpu, seq_len);
    IR_NODE_LINK_TO(mask_adaptive_xpu, seq_lod);
    IR_NODE_LINK_TO(mask_adaptive_xpu, max_seq_len);
    IR_NODE_LINK_TO(x, sequence_unpad);
    IR_NODE_LINK_TO(seq_len, sequence_unpad);
    IR_NODE_LINK_TO(sequence_unpad, x_vsl);
    IR_NODE_LINK_TO(x_vsl, multi_encoder_xpu);
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
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  std::vector<std::string> matmul_types{"matmul", "matmul_v2"};
  int found_subgraph_count = 0;
  for (auto& matmul_type : matmul_types) {
    found_subgraph_count += ApplyAdaptiveSeqlenPassV1(graph, matmul_type);
    found_subgraph_count += ApplyAdaptiveSeqlenPassV2(graph, matmul_type);
    found_subgraph_count += ApplyAdaptiveSeqlenPassV3(graph, matmul_type);
  }

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
