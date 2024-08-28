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

#include "paddle/fluid/framework/ir/xpu/fused_multi_transformer_int8_cachekv_layout_trans_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FusedMultiTransformerInt8FillConstantPattern : public PatternBase {
  FusedMultiTransformerInt8FillConstantPattern(PDPattern* pattern,
                                               const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fill_constant);
  PATTERN_DECL_NODE(fused_multi_transformer_int8);
  // declare variable node's name
  PATTERN_DECL_NODE(fill_constant_out);
};  // struct FusedMultiTransformerInt8FillConstantPattern

FusedMultiTransformerInt8FillConstantPattern::
    FusedMultiTransformerInt8FillConstantPattern(PDPattern* pattern,
                                                 const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* fill_constant = pattern->NewNode(fill_constant_repr())
                            ->assert_is_op("fill_constant")
                            ->assert_has_n_inputs(5)
                            ->assert_more([](Node* node) {
                              return node->Op()->GetAttrIfExists<std::string>(
                                         "friendly_device_type") != "xpu";
                            });
  auto* fill_constant_out = pattern->NewNode(fill_constant_out_repr())
                                ->assert_is_op_output("fill_constant", "Out");
  auto* fused_multi_transformer_int8 =
      pattern->NewNode(fused_multi_transformer_int8_repr())
          ->assert_is_op("fused_multi_transformer_int8");

  fill_constant->LinksTo({fill_constant_out});
  fused_multi_transformer_int8->LinksFrom({fill_constant_out});
}

struct FusedMultiTransformerIng8GatherPattern : public PatternBase {
  FusedMultiTransformerIng8GatherPattern(PDPattern* pattern,
                                         const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_multi_transformer_int8);
  PATTERN_DECL_NODE(gather);
  // declare variable node's name
  PATTERN_DECL_NODE(gather_in);
  PATTERN_DECL_NODE(gather_out);
};  // struct FusedMultiTransformerIng8GatherPattern

FusedMultiTransformerIng8GatherPattern::FusedMultiTransformerIng8GatherPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* gather_in =
      pattern->NewNode(gather_in_repr())->assert_is_op_input("gather", "X");
  auto* gather = pattern->NewNode(gather_repr())
                     ->assert_is_op("gather")
                     ->assert_more([](Node* node) {
                       return node->Op()->GetAttrIfExists<int>("axis") == 1;
                     });
  auto* gather_out =
      pattern->NewNode(gather_out_repr())->assert_is_op_output("gather", "Out");
  auto* fused_multi_transformer_int8 =
      pattern->NewNode(fused_multi_transformer_int8_repr())
          ->assert_is_op("fused_multi_transformer_int8");

  gather->LinksFrom({gather_in}).LinksTo({gather_out});
  fused_multi_transformer_int8->LinksFrom({gather_out});
}
}  // namespace patterns

void FusedMultiTransformerInt8CacheKVLayoutTransPass::FillConstantReshapePass(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerInt8FillConstantPattern pattern(
      gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FillConstantReshapePass";
    GET_IR_NODE(fused_multi_transformer_int8);
    GET_IR_NODE(fill_constant);
    GET_IR_NODE(fill_constant_out);
    auto cachekv_names = fused_multi_transformer_int8->Op()->Input("CacheKV");
    if (std::count(cachekv_names.begin(),
                   cachekv_names.end(),
                   fill_constant_out->Name()) == 0)
      return;

    auto fill_constant_input_names =
        fill_constant->Op()->Input("ShapeTensorList");
    auto fill_constant_trans_input_names =
        std::vector<std::string>{fill_constant_input_names[0],
                                 fill_constant_input_names[3],
                                 fill_constant_input_names[1],
                                 fill_constant_input_names[2],
                                 fill_constant_input_names[4]};
    fill_constant->Op()->SetInput("ShapeTensorList",
                                  fill_constant_trans_input_names);

    auto fill_constant_output_shape = fill_constant_out->Var()->GetShape();
    fill_constant_out->Var()->SetShape({fill_constant_output_shape[0],
                                        fill_constant_output_shape[3],
                                        fill_constant_output_shape[1],
                                        fill_constant_output_shape[2],
                                        fill_constant_output_shape[4]});

    fused_multi_transformer_int8->Op()->SetAttr("friendly_device_type",
                                                std::string("xpu"));
    fill_constant->Op()->SetAttr("friendly_device_type", std::string("xpu"));
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

int FusedMultiTransformerInt8CacheKVLayoutTransPass::
    CountFillConstantReshapePattern(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerInt8FillConstantPattern pattern(
      gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FillConstantReshapePass";
    GET_IR_NODE(fused_multi_transformer_int8);
    GET_IR_NODE(fill_constant);
    GET_IR_NODE(fill_constant_out);
    auto cachekv_names = fused_multi_transformer_int8->Op()->Input("CacheKV");
    if (std::count(cachekv_names.begin(),
                   cachekv_names.end(),
                   fill_constant_out->Name()) == 0)
      return;
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}

void FusedMultiTransformerInt8CacheKVLayoutTransPass::GatherReshapePass(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerIng8GatherPattern pattern(
      gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle GatherReshapePass";
    GET_IR_NODE(gather);
    GET_IR_NODE(fused_multi_transformer_int8);
    GET_IR_NODE(gather_in);
    GET_IR_NODE(gather_out);
    auto cachekv_names = fused_multi_transformer_int8->Op()->Input("CacheKV");
    if (std::count(cachekv_names.begin(),
                   cachekv_names.end(),
                   gather_out->Name()) == 0)
      return;

    auto gather_in_shape = gather_in->Var()->GetShape();
    auto gather_out_shape = gather_out->Var()->GetShape();
    gather_in->Var()->SetShape({gather_in_shape[0],
                                gather_in_shape[3],
                                gather_in_shape[1],
                                gather_in_shape[2],
                                gather_in_shape[4]});
    gather_out->Var()->SetShape({gather_out_shape[0],
                                 gather_out_shape[3],
                                 gather_out_shape[1],
                                 gather_out_shape[2],
                                 gather_out_shape[4]});
    gather->Op()->SetAttr("axis", 2);
    fused_multi_transformer_int8->Op()->SetAttr("friendly_device_type",
                                                std::string("xpu"));

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

int FusedMultiTransformerInt8CacheKVLayoutTransPass::CountGatherReshapePattern(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerIng8GatherPattern pattern(
      gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle GatherReshapePass";
    GET_IR_NODE(gather);
    GET_IR_NODE(fused_multi_transformer_int8);
    GET_IR_NODE(gather_in);
    GET_IR_NODE(gather_out);
    auto cachekv_names = fused_multi_transformer_int8->Op()->Input("CacheKV");
    if (std::count(cachekv_names.begin(),
                   cachekv_names.end(),
                   gather_out->Name()) == 0)
      return;
    found_subgraph_count++;
  };
  gpd(graph, handler);
  return found_subgraph_count;
}

void FusedMultiTransformerInt8CacheKVLayoutTransPass::ApplyImpl(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  if (!graph->IsMainGraph()) {
    VLOG(3) << "'fused_multi_transformer_cachekv_layout_pass' needs info in "
               "all graphs, so it should be applied in the main graph.";
    return;
  }
  Init(name_scope_, graph);
  int pattern_cnt0 = 0, pattern_cnt1 = 0;
  for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
    pattern_cnt0 += CountFillConstantReshapePattern(graph->GetSubGraph(i));
    pattern_cnt1 += CountGatherReshapePattern(graph->GetSubGraph(i));
  }
  if (pattern_cnt0 != 0 && pattern_cnt1 != 0 && pattern_cnt0 == pattern_cnt1) {
    for (size_t i = 0; i < graph->SubGraphsSize(); i++) {
      FillConstantReshapePass(graph->GetSubGraph(i));
      GatherReshapePass(graph->GetSubGraph(i));
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(
    fused_multi_transformer_int8_cachekv_layout_trans_pass,
    paddle::framework::ir::FusedMultiTransformerInt8CacheKVLayoutTransPass);
