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

#include <map>
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {
/*
fuse block in vis model to reformer_relative_pos_xpu op
------------------------------------------------------ */
/* support xpu roformer relative pos                    */
/*                    x ---------------                */
/*                    |    \             |              */
/*                    |     \            |              */
/*                  split    shape       |              */
/*                 /  |        \         |              */
/*                /   |         \        |              */
/*               |  scale      slice     |              */
/*                \   |         /  \     |              */
/*                 \  |        /    \    |              */
/*                  concat  slice  slice |              */
/*                    |      /        \  |              */
/*                    |     /          \ |              */
/*             elementwise_mul     elementwise_mul      */
/*                    |           /                     */
/*                    |          /                      */
/*                elementwise_add                       */
/*                    |                                 */
/*                    |                                 */
/*                   out                                */
/*-------------------------------------------*/
/* After the pass apply:                     */
/*                    x                      */
/*          cos_emb   |   sin_emb            */
/*                 \  |  /                   */
/*          xpu_roformer_relative            */
/*                    |                      */
/*                    |                      */
/*                   out                     */
/*-------------------------------------------*/

struct RoformerRelativePosXPUPattern : public PatternBase {
  RoformerRelativePosXPUPattern(PDPattern* pattern,
                                const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(split);
  PATTERN_DECL_NODE(scale);
  PATTERN_DECL_NODE(concat);
  PATTERN_DECL_NODE(mul1);

  PATTERN_DECL_NODE(shape);
  PATTERN_DECL_NODE(slice1);
  PATTERN_DECL_NODE(slice_sin);
  PATTERN_DECL_NODE(slice_cos);

  PATTERN_DECL_NODE(mul2);
  PATTERN_DECL_NODE(add);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(sin_emb);
  PATTERN_DECL_NODE(cos_emb);
  PATTERN_DECL_NODE(split_out1);
  PATTERN_DECL_NODE(split_out2);
  PATTERN_DECL_NODE(scale_out);
  PATTERN_DECL_NODE(concat_out);
  PATTERN_DECL_NODE(mul1_out);
  PATTERN_DECL_NODE(shape_out);
  PATTERN_DECL_NODE(slice1_out);
  PATTERN_DECL_NODE(slice_sin_out);
  PATTERN_DECL_NODE(slice_cos_out);
  PATTERN_DECL_NODE(mul2_out);
  PATTERN_DECL_NODE(add_out);
};

RoformerRelativePosXPUPattern::RoformerRelativePosXPUPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input("split", "X")
                ->assert_is_op_input("elementwise_mul", "X")
                ->assert_is_op_input("shape", "Input")
                ->AsInput();

  auto* split = pattern->NewNode(split_repr())
                    ->assert_is_op("split")
                    ->assert_op_attr<int>("axis", 3)
                    ->assert_op_attr<int>("num", 2);  // do we really need it

  auto* split_out1 = pattern->NewNode(split_out1_repr())
                         ->assert_is_op_input("scale", "X")
                         ->assert_is_op_nth_output("split", "Out", 1);
  auto* split_out2 = pattern->NewNode(split_out2_repr())
                         ->assert_is_op_nth_input("concat", "X", 1)
                         ->assert_is_op_nth_output("split", "Out", 0);
  split->LinksFrom({x}).LinksTo({split_out1, split_out2});

  auto* scale = pattern->NewNode(scale_repr())
                    ->assert_is_op("scale")
                    ->assert_more([&](Node* node) {
                      auto* op_desc = node->Op();
                      auto scale = op_desc->GetAttrIfExists<float>("scale");
                      return (std::fabs(scale + 1.0) < 1e-5);
                    });
  auto* scale_out = pattern->NewNode(scale_out_repr())
                        ->assert_is_op_input("concat", "X")
                        ->assert_is_op_output("scale", "Out");
  scale->LinksFrom({split_out1}).LinksTo({scale_out});
  auto* concat = pattern->NewNode(concat_repr())->assert_is_op("concat");
  auto* concat_out = pattern->NewNode(concat_out_repr())
                         ->assert_is_op_input("elementwise_mul", "X")
                         ->assert_is_op_output("concat", "Out");
  concat->LinksFrom({scale_out, split_out2}).LinksTo({concat_out});
  auto* shape = pattern->NewNode(shape_repr())->assert_is_op("shape");
  auto* shape_out = pattern->NewNode(shape_out_repr())
                        ->assert_is_op_input("slice", "Input")
                        ->assert_is_op_output("shape", "Out");
  shape->LinksFrom({x}).LinksTo({shape_out});
  auto* slice1 = pattern->NewNode(slice1_repr())->assert_is_op("slice");
  auto* slice1_out = pattern->NewNode(slice1_out_repr())
                         ->assert_is_op_input("slice", "EndsTensorList")
                         ->assert_is_op_output("slice", "Out");
  slice1->LinksFrom({shape_out}).LinksTo({slice1_out});
  auto* sin_emb = pattern->NewNode(sin_emb_repr())
                      ->assert_is_op_input("slice", "Input")
                      ->AsInput();
  auto* cos_emb = pattern->NewNode(cos_emb_repr())
                      ->assert_is_op_input("slice", "Input")
                      ->AsInput();
  auto* slice_sin = pattern->NewNode(slice_sin_repr())->assert_is_op("slice");
  auto* slice_sin_out = pattern->NewNode(slice_sin_out_repr())
                            ->assert_is_op_input("elementwise_mul", "Y")
                            ->assert_is_op_output("slice", "Out");
  slice_sin->LinksFrom({sin_emb, slice1_out}).LinksTo({slice_sin_out});
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("elementwise_mul");
  auto* mul1_out = pattern->NewNode(mul1_out_repr())
                       ->assert_is_op_input("elementwise_add", "Y")
                       ->assert_is_op_output("elementwise_mul", "Out");
  mul1->LinksFrom({concat_out, slice_sin_out}).LinksTo({mul1_out});
  auto* add = pattern->NewNode(add_repr())->assert_is_op("elementwise_add");
  auto* add_out = pattern->NewNode(add_out_repr())
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->AsOutput();
  auto* slice_cos = pattern->NewNode(slice_cos_repr())->assert_is_op("slice");
  auto* slice_cos_out = pattern->NewNode(slice_cos_out_repr())
                            ->assert_is_op_input("elementwise_mul", "Y")
                            ->assert_is_op_output("slice", "Out");
  slice_cos->LinksFrom({cos_emb, slice1_out}).LinksTo({slice_cos_out});
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_op("elementwise_mul");
  auto* mul2_out = pattern->NewNode(mul2_out_repr())
                       ->assert_is_op_input("elementwise_add", "X")
                       ->assert_is_op_output("elementwise_mul", "Out");
  mul2->LinksFrom({x, slice_cos_out}).LinksTo({mul2_out});
  add->LinksFrom({mul2_out, mul1_out}).LinksTo({add_out});
}

}  // namespace patterns

class RoformerRelativePosFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"roformer_relative_pos_fuse_pass"};
};

void RoformerRelativePosFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::RoformerRelativePosXPUPattern pattern(gpd.mutable_pattern(),
                                                  name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle RoformerRelativePosFusePass fuse";
    /* declare operator node's name */
    // declare variable node's name
    GET_IR_NODE(split);
    GET_IR_NODE(scale);
    GET_IR_NODE(concat);
    GET_IR_NODE(mul1);
    GET_IR_NODE(shape);
    GET_IR_NODE(slice1);
    GET_IR_NODE(slice_sin);
    GET_IR_NODE(slice_cos);
    GET_IR_NODE(mul2);
    GET_IR_NODE(add);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(sin_emb);
    GET_IR_NODE(cos_emb);
    GET_IR_NODE(split_out1);
    GET_IR_NODE(split_out2);
    GET_IR_NODE(scale_out);
    GET_IR_NODE(concat_out);
    GET_IR_NODE(mul1_out);
    GET_IR_NODE(shape_out);
    GET_IR_NODE(slice1_out);
    GET_IR_NODE(slice_sin_out);
    GET_IR_NODE(slice_cos_out);
    GET_IR_NODE(mul2_out);
    GET_IR_NODE(add_out);
    auto* block = add->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    // Generate roformer_relative_embedding_xpu fused op
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("roformer_relative_embedding_xpu");
    // set attrs for fused op
    fused_op_desc.SetInput("x", {x->Name()});
    fused_op_desc.SetInput("sin_emb", {sin_emb->Name()});
    fused_op_desc.SetInput("cos_emb", {cos_emb->Name()});

    fused_op_desc.SetOutput("out", {add_out->Name()});
    fused_op_desc.SetAttr("max_pos_len",
                          static_cast<int>(cos_emb->Var()->GetShape()[2]));

    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(x, fused_op);
    IR_NODE_LINK_TO(sin_emb, fused_op);
    IR_NODE_LINK_TO(cos_emb, fused_op);
    IR_NODE_LINK_TO(fused_op, add_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {split,
                                                    scale,
                                                    concat,
                                                    mul1,
                                                    shape,
                                                    slice1,
                                                    slice_sin,
                                                    slice_cos,
                                                    mul2,
                                                    add,
                                                    split_out1,
                                                    split_out2,
                                                    scale_out,
                                                    concat_out,
                                                    shape_out,
                                                    slice1_out,
                                                    slice_sin_out,
                                                    slice_cos_out,
                                                    mul2_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);

  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(roformer_relative_pos_fuse_pass,
              paddle::framework::ir::RoformerRelativePosFusePass);

REGISTER_PASS_CAPABILITY(roformer_relative_pos_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "roformer_relative_embedding_xpu", 0));
