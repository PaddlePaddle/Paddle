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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
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
fuse block in yolo-like model to yolo_box_xpu op
------------------------------------------------------
sub block:
                      x
                /     |     \
               /      |      \
              /       |       \
           slice    slice    slice
             |        |        |
             |        |        |
           ew_mul   ew_mul     |
             |        |        |
             |        |        |
           ew_sub   ew_pow     |
             |        |        |
             |        |        |
           ew_add   ew_mul_2   |
             |        |        |
             |        |        |
           ew_mul_2   |        |
              \       |       /
               \      |      /
                \     |     /
                    concat
                      |
                      y
------------------------------------------------------
After the pass is applied:
                              x
     grid[left_ew_add_y]      |     offset[left_ew_sub_y]
                       \      |    /
                        \     |   /
stride[left_ew_mul_2_y] -- yolo_box_xpu --- anchor_grid[mid_ew_mul_2_y]
                              |    \
                              |     \
                              |      \
                              y      y_max
*/
struct YoloBoxXPUPattern : public PatternBase {
  YoloBoxXPUPattern(PDPattern* pattern,
                    const std::string& name_scope,
                    bool with_left_ew_sub_);
  // declare operator node's name
  PATTERN_DECL_NODE(left_slice);
  PATTERN_DECL_NODE(mid_slice);
  PATTERN_DECL_NODE(right_slice);
  PATTERN_DECL_NODE(left_ew_mul);
  PATTERN_DECL_NODE(left_ew_sub);
  PATTERN_DECL_NODE(left_ew_add);
  PATTERN_DECL_NODE(left_ew_mul_2);
  PATTERN_DECL_NODE(mid_ew_mul);
  PATTERN_DECL_NODE(mid_ew_pow);
  PATTERN_DECL_NODE(mid_ew_mul_2);
  PATTERN_DECL_NODE(concat);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(left_slice_out);
  PATTERN_DECL_NODE(left_ew_mul_out);
  PATTERN_DECL_NODE(left_ew_sub_y);
  PATTERN_DECL_NODE(left_ew_sub_out);
  PATTERN_DECL_NODE(left_ew_add_y);
  PATTERN_DECL_NODE(left_ew_add_out);
  PATTERN_DECL_NODE(left_ew_mul_2_y);
  PATTERN_DECL_NODE(left_ew_mul_2_out);
  PATTERN_DECL_NODE(mid_slice_out);
  PATTERN_DECL_NODE(mid_ew_mul_out);
  PATTERN_DECL_NODE(mid_ew_pow_out);
  PATTERN_DECL_NODE(mid_ew_mul_2_y);
  PATTERN_DECL_NODE(mid_ew_mul_2_out);
  PATTERN_DECL_NODE(right_slice_out);
  PATTERN_DECL_NODE(concat_out);

 private:
  bool with_left_ew_sub_{true};
};

YoloBoxXPUPattern::YoloBoxXPUPattern(PDPattern* pattern,
                                     const std::string& name_scope,
                                     bool with_left_ew_sub)
    : PatternBase(pattern, name_scope, name_scope),
      with_left_ew_sub_(with_left_ew_sub) {
  auto x = pattern->NewNode(x_repr())
               ->assert_is_op_input("strided_slice", "Input")
               ->assert_has_n_outputs(3)
               ->assert_more([](Node* node) {
                 auto dim_size = node->Var()->GetShape().size();
                 return node->Var()->GetShape()[dim_size - 1] > 4;
               });
  // left silce pattern
  auto* left_slice =
      pattern->NewNode(left_slice_repr())->assert_is_op("strided_slice");
  auto* left_slice_out = pattern->NewNode(left_slice_out_repr())
                             ->assert_is_op_output("strided_slice", "Out")
                             ->assert_is_op_input("elementwise_mul", "X");
  left_slice->LinksFrom({x}).LinksTo({left_slice_out});
  auto* left_ew_mul =
      pattern->NewNode(left_ew_mul_repr())->assert_is_op("elementwise_mul");
  auto* left_ew_mul_out = pattern->NewNode(left_ew_mul_out_repr())
                              ->assert_is_op_output("elementwise_mul", "Out");
  left_ew_mul->LinksFrom({left_slice_out}).LinksTo({left_ew_mul_out});
  PDNode* left_ew_sub = nullptr;
  PDNode* left_ew_sub_y = nullptr;
  PDNode* left_ew_sub_out = nullptr;
  if (with_left_ew_sub_) {
    left_ew_mul_out->assert_is_op_input("elementwise_sub", "X");
    left_ew_sub =
        pattern->NewNode(left_ew_sub_repr())->assert_is_op("elementwise_sub");
    left_ew_sub_y = pattern->NewNode(left_ew_sub_y_repr())
                        ->assert_is_op_input("elementwise_sub", "Y");
    left_ew_sub_out = pattern->NewNode(left_ew_sub_out_repr())
                          ->assert_is_op_output("elementwise_sub", "Out");
    left_ew_sub->LinksFrom({left_ew_mul_out}).LinksTo({left_ew_sub_out});
  } else {
    left_ew_sub_out = left_ew_mul_out;
  }
  left_ew_sub_out->assert_is_op_input("elementwise_add", "X");
  auto* left_ew_add =
      pattern->NewNode(left_ew_add_repr())->assert_is_op("elementwise_add");
  auto* left_ew_add_y = pattern->NewNode(left_ew_add_y_repr())
                            ->assert_is_op_input("elementwise_add", "Y");
  auto* left_ew_add_out = pattern->NewNode(left_ew_add_out_repr())
                              ->assert_is_op_output("elementwise_add", "Out")
                              ->assert_is_op_input("elementwise_mul", "X");
  left_ew_add->LinksFrom({left_ew_sub_out}).LinksTo({left_ew_add_out});
  auto* left_ew_mul_2 =
      pattern->NewNode(left_ew_mul_2_repr())->assert_is_op("elementwise_mul");
  auto* left_ew_mul_2_y = pattern->NewNode(left_ew_mul_2_y_repr())
                              ->assert_is_op_input("elementwise_mul", "Y");
  auto* left_ew_mul_2_out = pattern->NewNode(left_ew_mul_2_out_repr())
                                ->assert_is_op_output("elementwise_mul", "Out")
                                ->assert_is_op_input("concat", "X");
  left_ew_mul_2->LinksFrom({left_ew_add_out}).LinksTo({left_ew_mul_2_out});
  // mid slice pattern
  auto* mid_slice =
      pattern->NewNode(mid_slice_repr())->assert_is_op("strided_slice");
  auto* mid_slice_out = pattern->NewNode(mid_slice_out_repr())
                            ->assert_is_op_output("strided_slice", "Out")
                            ->assert_is_op_input("elementwise_mul", "X");
  mid_slice->LinksFrom({x}).LinksTo({mid_slice_out});
  auto* mid_ew_mul =
      pattern->NewNode(mid_ew_mul_repr())->assert_is_op("elementwise_mul");
  auto* mid_ew_mul_out = pattern->NewNode(mid_ew_mul_out_repr())
                             ->assert_is_op_output("elementwise_mul", "Out")
                             ->assert_is_op_input("elementwise_pow", "X");
  mid_ew_mul->LinksFrom({mid_slice_out}).LinksTo({mid_ew_mul_out});
  auto* mid_ew_pow =
      pattern->NewNode(mid_ew_pow_repr())->assert_is_op("elementwise_mul");
  auto* mid_ew_pow_out = pattern->NewNode(mid_ew_pow_out_repr())
                             ->assert_is_op_output("elementwise_pow", "Out")
                             ->assert_is_op_input("elementwise_mul", "X");
  mid_ew_mul->LinksFrom({mid_ew_mul_out}).LinksTo({mid_ew_pow_out});
  auto* mid_ew_mul_2 =
      pattern->NewNode(mid_ew_mul_2_repr())->assert_is_op("elementwise_mul");
  auto* mid_ew_mul_2_y = pattern->NewNode(mid_ew_mul_2_y_repr())
                             ->assert_is_op_input("elementwise_mul", "Y");
  auto* mid_ew_mul_2_out = pattern->NewNode(mid_ew_mul_2_out_repr())
                               ->assert_is_op_output("elementwise_mul", "Out")
                               ->assert_is_op_input("concat", "X");
  mid_ew_mul_2->LinksFrom({mid_ew_pow_out}).LinksTo({mid_ew_mul_2_out});
  // right slice pattern
  auto* right_slice =
      pattern->NewNode(right_slice_repr())->assert_is_op("strided_slice");
  auto* right_slice_out = pattern->NewNode(right_slice_out_repr())
                              ->assert_is_op_output("strided_slice", "Out")
                              ->assert_is_op_input("concat", "X");
  right_slice->LinksFrom({x}).LinksTo({right_slice_out});
  // concat
  auto* concat = pattern->NewNode(concat_repr())->assert_is_op("concat");
  auto* concat_out = pattern->NewNode(concat_out_repr())
                         ->assert_is_op_output("concat", "Out")
                         ->AsOutput();
  concat->LinksFrom({left_ew_mul_2_out, mid_ew_mul_2_out, right_slice_out})
      .LinksTo({concat_out});
}

}  // namespace patterns

class YoloBoxXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  int ApplyImpl(ir::Graph* graph, bool with_left_ew_sub) const;

  const std::string name_scope_{"yolo_box_xpu_fuse_pass"};
};

void YoloBoxXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  int found_subgraph_count = 0;
  for (auto with_left_ew_sub : {true, false}) {
    found_subgraph_count += ApplyImpl(graph, with_left_ew_sub);
  }
  AddStatis(found_subgraph_count);
}

int YoloBoxXPUFusePass::ApplyImpl(ir::Graph* graph,
                                  bool with_left_ew_sub) const {
  GraphPatternDetector gpd;
  patterns::YoloBoxXPUPattern pattern(
      gpd.mutable_pattern(), name_scope_, with_left_ew_sub);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle YoloBoxXPUFusePass fuse";
    /* declare operator node's name */
    // declare operator node's name
    GET_IR_NODE(left_slice);
    GET_IR_NODE(mid_slice);
    GET_IR_NODE(right_slice);
    GET_IR_NODE(left_ew_mul);
    GET_IR_NODE(left_ew_sub);
    GET_IR_NODE(left_ew_add);
    GET_IR_NODE(left_ew_mul_2);
    GET_IR_NODE(mid_ew_mul);
    GET_IR_NODE(mid_ew_pow);
    GET_IR_NODE(mid_ew_mul_2);
    GET_IR_NODE(concat);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(left_slice_out);
    GET_IR_NODE(left_ew_mul_out);
    GET_IR_NODE(left_ew_sub_y);
    GET_IR_NODE(left_ew_sub_out);
    GET_IR_NODE(left_ew_add_y);
    GET_IR_NODE(left_ew_add_out);
    GET_IR_NODE(left_ew_mul_2_y);
    GET_IR_NODE(left_ew_mul_2_out);
    GET_IR_NODE(mid_slice_out);
    GET_IR_NODE(mid_ew_mul_out);
    GET_IR_NODE(mid_ew_pow_out);
    GET_IR_NODE(mid_ew_mul_2_y);
    GET_IR_NODE(mid_ew_mul_2_out);
    GET_IR_NODE(right_slice_out);
    GET_IR_NODE(concat_out);
    auto* block = concat->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

    std::string fused_op_out_name;
    fused_op_out_name = concat_out->Name();
    std::string fused_op_out_max_name = fused_op_out_name + "_max";
    VarDesc fused_op_out_max_desc(fused_op_out_max_name);
    Node* fused_op_out_max = graph->CreateVarNode(&fused_op_out_max_desc);
    // Generate yolo_box_xpu fused op
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("yolo_box_xpu");
    // set attrs for fused op
    fused_op_desc.SetInput("x", {x->Name()});
    fused_op_desc.SetInput("grid", {left_ew_add_y->Name()});
    fused_op_desc.SetInput("stride", {left_ew_mul_2_y->Name()});
    if (with_left_ew_sub) {
      fused_op_desc.SetInput("offset", {left_ew_sub_y->Name()});
    }
    fused_op_desc.SetInput("anchor_grid", {mid_ew_mul_2_y->Name()});
    fused_op_desc.SetOutput("y", {concat_out->Name()});
    fused_op_desc.SetOutput("y_max", {fused_op_out_max_name});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(x, fused_op);
    IR_NODE_LINK_TO(left_ew_add_y, fused_op);
    IR_NODE_LINK_TO(left_ew_mul_2_y, fused_op);
    if (with_left_ew_sub) {
      IR_NODE_LINK_TO(left_ew_sub_y, fused_op);
    }
    IR_NODE_LINK_TO(mid_ew_mul_2_y, fused_op);
    IR_NODE_LINK_TO(fused_op, concat_out);
    IR_NODE_LINK_TO(fused_op, fused_op_out_max);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {left_slice,
                                                    mid_slice,
                                                    right_slice,
                                                    left_ew_mul,
                                                    left_ew_add,
                                                    left_ew_mul_2,
                                                    mid_ew_mul,
                                                    mid_ew_pow,
                                                    mid_ew_mul_2,
                                                    concat,
                                                    left_slice_out,
                                                    left_ew_mul_out,
                                                    left_ew_add_out,
                                                    left_ew_mul_2_out,
                                                    mid_slice_out,
                                                    mid_ew_mul_out,
                                                    mid_ew_pow_out,
                                                    mid_ew_mul_2_out,
                                                    right_slice_out};
    if (with_left_ew_sub) {
      delete_nodes.insert(left_ew_sub, left_ew_sub_out);
    }
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(yolo_box_xpu_fuse_pass,
              paddle::framework::ir::YoloBoxXPUFusePass);

REGISTER_PASS_CAPABILITY(yolo_box_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "yolo_box_xpu", 0));
