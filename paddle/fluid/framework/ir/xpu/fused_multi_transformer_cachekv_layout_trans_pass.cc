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

#include "paddle/fluid/framework/ir/xpu/fused_multi_transformer_cachekv_layout_trans_pass.h"
#include <string>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace {
const int fill_constant_shape_tensor_list_names_size = 5;
}

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FusedMultiTransformerFillConstantPattern : public PatternBase {
  FusedMultiTransformerFillConstantPattern(PDPattern* pattern,
                                           const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_multi_transformer);
  PATTERN_DECL_NODE(fill_constant_reduce);
  // declare variable node's name
  PATTERN_DECL_NODE(fill_constant_reduce_out);
};  // struct FusedMultiTransformerFillConstantPattern

FusedMultiTransformerFillConstantPattern::
    FusedMultiTransformerFillConstantPattern(PDPattern* pattern,
                                             const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  // fill_constant_reduce
  auto* fill_constant_reduce = pattern->NewNode(fill_constant_reduce_repr())
                                   ->assert_is_op("fill_constant")
                                   ->assert_has_n_inputs(5);
  auto* fill_constant_reduce_out =
      pattern->NewNode(fill_constant_reduce_out_repr())
          ->assert_is_op_output("fill_constant", "Out");
  fill_constant_reduce->LinksTo({fill_constant_reduce_out});

  // fused_multi_trans_former
  auto* fused_multi_trans_former =
      pattern->NewNode(fused_multi_transformer_repr())
          ->assert_is_op("fused_multi_transformer");
  fused_multi_trans_former->LinksFrom({fill_constant_reduce_out});
}

struct FusedMultiTransformerGatherPattern : public PatternBase {
  FusedMultiTransformerGatherPattern(PDPattern* pattern,
                                     const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_multi_transformer);
  PATTERN_DECL_NODE(gather);
  // declare variable node's name
  PATTERN_DECL_NODE(gather_in);
  PATTERN_DECL_NODE(gather_out);
};  // struct FusedMultiTransformerGatherPattern

FusedMultiTransformerGatherPattern::FusedMultiTransformerGatherPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* fused_multi_transformer =
      pattern->NewNode(fused_multi_transformer_repr())
          ->assert_is_op("fused_multi_transformer");
  auto* gather_out =
      pattern->NewNode(gather_out_repr())->assert_is_op_output("gather", "Out");
  fused_multi_transformer->LinksFrom({gather_out});

  auto* gather = pattern->NewNode(gather_repr())->assert_is_op("gather");
  auto* gather_in =
      pattern->NewNode(gather_in_repr())->assert_is_op_input("gather", "X");
  gather->LinksFrom({gather_in}).LinksTo({gather_out});
}
}  // namespace patterns

void FusedMultiTransformerCacheKVLayoutTransPass::ApplyImpl(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FillConstantReshapePass(graph);
  GatherReshapePass(graph);
}

void FusedMultiTransformerCacheKVLayoutTransPass::FillConstantReshapePass(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerFillConstantPattern pattern(
      gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FillConstantReshapePass";
    GET_IR_NODE(fused_multi_transformer);
    GET_IR_NODE(fill_constant_reduce);
    GET_IR_NODE(fill_constant_reduce_out);
    auto fill_constant_reduce_input_names =
        fill_constant_reduce->Op()->Input("ShapeTensorList");
    auto fill_constant_reduce_trans_input_names =
        std::vector<std::string>{fill_constant_reduce_input_names[0],
                                 fill_constant_reduce_input_names[3],
                                 fill_constant_reduce_input_names[1],
                                 fill_constant_reduce_input_names[2],
                                 fill_constant_reduce_input_names[4]};
    fill_constant_reduce->Op()->SetInput(
        "ShapeTensorList", fill_constant_reduce_trans_input_names);

    auto fill_constant_reduce_output_shape =
        fill_constant_reduce_out->Var()->GetShape();
    fill_constant_reduce_out->Var()->SetShape(
        {fill_constant_reduce_output_shape[0],
         fill_constant_reduce_output_shape[3],
         fill_constant_reduce_output_shape[1],
         fill_constant_reduce_output_shape[2],
         fill_constant_reduce_output_shape[4]});

    fused_multi_transformer->Op()->SetAttr("friendly_device_type",
                                           std::string("xpu"));
    ++found_subgraph_count;
  };
  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void FusedMultiTransformerCacheKVLayoutTransPass::GatherReshapePass(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerGatherPattern pattern(gpd.mutable_pattern(),
                                                       name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle GatherReshapePass";
    GET_IR_NODE(fused_multi_transformer);
    GET_IR_NODE(gather_in);
    GET_IR_NODE(gather_out);

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
    fused_multi_transformer->Op()->SetAttr("friendly_device_type",
                                           std::string("xpu"));
    ++found_subgraph_count;
  };
  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(
    fused_multi_transformer_cachekv_layout_trans_pass,
    paddle::framework::ir::FusedMultiTransformerCacheKVLayoutTransPass);
