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

#include "paddle/fluid/framework/ir/xpu/fold_interp_outsize_fuse_pass.h"
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

struct InterpOutsizeFusePattern : public PatternBase {
  InterpOutsizeFusePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(shape);
  PATTERN_DECL_NODE(cast1);
  PATTERN_DECL_NODE(slice);
  PATTERN_DECL_NODE(concat);
  PATTERN_DECL_NODE(split);
  PATTERN_DECL_NODE(cast2);
  PATTERN_DECL_NODE(bilinear_interp);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(shape_out);
  PATTERN_DECL_NODE(cast1_out);
  PATTERN_DECL_NODE(slice_out);
  PATTERN_DECL_NODE(concat_y);
  PATTERN_DECL_NODE(concat_out);
  PATTERN_DECL_NODE(split_out_0);
  PATTERN_DECL_NODE(split_out_1);
  PATTERN_DECL_NODE(cast2_out);
};

InterpOutsizeFusePattern::InterpOutsizeFusePattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input("shape", "Input")
                ->assert_is_op_input("bilinear_interp_v2", "X");
  auto* shape = pattern->NewNode(shape_repr())->assert_is_op("shape");
  auto* shape_out = pattern->NewNode(shape_out_repr())
                        ->assert_is_op_output("shape", "Out")
                        ->assert_is_op_input("cast", "X");
  shape->LinksFrom({x}).LinksTo({shape_out});
  auto* cast1 = pattern->NewNode(cast1_repr())
                    ->assert_is_op("cast")
                    ->assert_more([&](Node* node) {
                      auto* op_desc = node->Op();
                      return op_desc->GetAttrIfExists<int>("in_dtype") == 2 &&
                             op_desc->GetAttrIfExists<int>("out_dtype") == 3;
                    });
  auto* cast1_out = pattern->NewNode(cast1_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("slice", "Input");
  cast1->LinksFrom({shape_out}).LinksTo({cast1_out});
  auto* slice =
      pattern->NewNode(slice_repr())
          ->assert_is_op("slice")
          ->assert_more([&](Node* node) {
            auto* op_desc = node->Op();
            return op_desc->GetAttrIfExists<std::vector<int>>("axes") ==
                       std::vector<int>{0} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("starts") ==
                       std::vector<int>{0} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("ends") ==
                       std::vector<int>{2};
          });
  auto* slice_out = pattern->NewNode(slice_out_repr())
                        ->assert_is_op_output("slice", "Out")
                        ->assert_is_op_nth_input("concat", "X", 0);
  slice->LinksFrom({cast1_out}).LinksTo({slice_out});
  auto* concat = pattern->NewNode(concat_repr())
                     ->assert_is_op("concat")
                     ->assert_more([&](Node* node) {
                       auto* op_desc = node->Op();
                       return op_desc->GetAttrIfExists<int>("axis") == 0;
                     });
  auto* concat_y = pattern->NewNode(concat_y_repr())
                       ->assert_is_op_nth_input("concat", "X", 1)
                       ->assert_is_persistable_var();
  auto* concat_out = pattern->NewNode(concat_out_repr())
                         ->assert_is_op_output("concat", "Out")
                         ->assert_is_op_input("split", "X");
  concat->LinksFrom({slice_out, concat_y}).LinksTo({concat_out});
  auto* split = pattern->NewNode(split_repr())
                    ->assert_is_op("split")
                    ->assert_more([&](Node* node) {
                      auto* op_desc = node->Op();
                      return op_desc->GetAttrIfExists<int>("axis") == 0 &&
                             (op_desc->GetAttrIfExists<std::vector<int>>(
                                  "sections") == std::vector<int>{2, 2} ||
                              op_desc->GetAttrIfExists<int>("num") == 2);
                    });
  auto* split_out_0 = pattern->NewNode(split_out_0_repr())
                          ->assert_is_op_nth_output("split", "Out", 0);
  auto* split_out_1 = pattern->NewNode(split_out_1_repr())
                          ->assert_is_op_nth_output("split", "Out", 1)
                          ->assert_is_op_input("cast", "X");
  split->LinksFrom({concat_out}).LinksTo({split_out_0, split_out_1});
  auto* cast2 = pattern->NewNode(cast2_repr())
                    ->assert_is_op("cast")
                    ->assert_more([&](Node* node) {
                      auto* op_desc = node->Op();
                      return op_desc->GetAttrIfExists<int>("in_dtype") == 3 &&
                             op_desc->GetAttrIfExists<int>("out_dtype") == 2;
                    });
  auto* cast2_out = pattern->NewNode(cast2_out_repr())
                        ->assert_is_op_output("cast", "Out")
                        ->assert_is_op_input("bilinear_interp_v2", "OutSize");
  cast2->LinksFrom({split_out_1}).LinksTo({cast2_out});
  auto* bilinear_interp = pattern->NewNode(bilinear_interp_repr())
                              ->assert_is_op("bilinear_interp_v2");
  bilinear_interp->LinksFrom({x, cast2_out});
}

}  // namespace patterns

void FoldInterpOutsizeFusePass::FoldInterpOutsize(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::InterpOutsizeFusePattern pattern(gpd.mutable_pattern(),
                                             name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DetectorFuse";
    /* declare operator node's name */
    GET_IR_NODE(shape);
    GET_IR_NODE(cast1);
    GET_IR_NODE(slice);
    GET_IR_NODE(concat);
    GET_IR_NODE(split);
    GET_IR_NODE(cast2);
    GET_IR_NODE(bilinear_interp);
    /* declare variable node's name*/
    GET_IR_NODE(x);
    GET_IR_NODE(shape_out);
    GET_IR_NODE(cast1_out);
    GET_IR_NODE(slice_out);
    GET_IR_NODE(concat_y);
    GET_IR_NODE(concat_out);
    GET_IR_NODE(split_out_0);
    GET_IR_NODE(split_out_1);
    GET_IR_NODE(cast2_out);

    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));

    auto* concat_y_t =
        scope->GetVar(concat_y->Name())->GetMutable<phi::DenseTensor>();
    // concat_y int64 --> int32
    auto tensor_type = concat_y_t->dtype();
    if (tensor_type == phi::DataType::INT64) {
      CastToInt32(concat_y_t, nullptr);
    }
    bilinear_interp->Op()->RenameInput(cast2_out->Name(), concat_y->Name());
    IR_NODE_UNLINK(x, shape);
    IR_NODE_UNLINK(cast2_out, bilinear_interp);
    IR_NODE_LINK_TO(concat_y, bilinear_interp);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {shape,
                                                    cast1,
                                                    slice,
                                                    concat,
                                                    split,
                                                    cast2,
                                                    shape_out,
                                                    cast1_out,
                                                    slice_out,
                                                    concat_out,
                                                    split_out_0,
                                                    split_out_1,
                                                    cast2_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void FoldInterpOutsizeFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FoldInterpOutsize(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fold_interp_outsize_fuse_pass,
              paddle::framework::ir::FoldInterpOutsizeFusePass);

REGISTER_PASS_CAPABILITY(fold_interp_outsize_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "shape", 0));
