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
fuse block in vis model to sine_pos_xpu op
------------------------------------------------------
sub block:
    x         y
     \       /
      \     /
       \   /
        mul
       /  \
      /    \
     /      \
  slice    slice
    |        |
    |        |
   sin       cos
    \        /
     \      /
      \    /
      stack
        |
        |
      flatten
        |
       out
------------------------------------------------------
After the pass is applied:
    x         y
     \       /
      \     /
       \   /
    sine_pos_xpu
         |
         |
        out
*/

struct SinePosXPUPattern : public PatternBase {
  SinePosXPUPattern(PDPattern* pattern, const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(ew_mul);
  PATTERN_DECL_NODE(slice1);
  PATTERN_DECL_NODE(slice2);
  PATTERN_DECL_NODE(sin);
  PATTERN_DECL_NODE(cos);
  PATTERN_DECL_NODE(stack);
  PATTERN_DECL_NODE(flatten);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(y);
  PATTERN_DECL_NODE(ew_mul_out);
  PATTERN_DECL_NODE(slice1_out);
  PATTERN_DECL_NODE(slice2_out);
  PATTERN_DECL_NODE(sin_out);
  PATTERN_DECL_NODE(cos_out);
  PATTERN_DECL_NODE(stack_out);
  PATTERN_DECL_NODE(flatten_out);
};

SinePosXPUPattern::SinePosXPUPattern(PDPattern* pattern,
                                     const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto x = pattern->NewNode(x_repr())
               ->assert_is_op_input("elementwise_mul", "X")
               ->assert_more([&](Node* node) {
                 auto x_shape = node->Var()->GetShape();
                 size_t x_rank = x_shape.size();
                 return x_rank == 3 && x_shape.back() == 1;
               });
  auto y = pattern->NewNode(y_repr())
               ->assert_is_op_input("elementwise_mul", "Y")
               ->assert_more([&](Node* node) {
                 auto x_shape = node->Var()->GetShape();
                 size_t x_rank = x_shape.size();
                 return x_rank == 1 && x_shape[0] % 2 == 0;
               });
  auto* ew_mul = pattern->NewNode(ew_mul_repr())
                     ->assert_is_op("elementwise_mul")
                     ->assert_more([&](Node* node) {
                       auto* op_desc = node->Op();
                       return op_desc->GetAttrIfExists<int>("axis") == -1;
                     });
  auto* ew_mul_out = pattern->NewNode(ew_mul_out_repr())
                         ->assert_is_op_output("elementwise_mul", "Out")
                         ->assert_is_op_input("strided_slice", "Input");
  ew_mul->LinksFrom({x, y}).LinksTo({ew_mul_out});
  auto* slice1 =
      pattern->NewNode(slice1_repr())
          ->assert_is_op("strided_slice")
          ->assert_more([&](Node* node) {
            auto* op_desc = node->Op();
            return op_desc->GetAttrIfExists<std::vector<int>>("axes") ==
                       std::vector<int>{2} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("starts") ==
                       std::vector<int>{0} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("strides") ==
                       std::vector<int>{2};
          });
  auto* slice1_out = pattern->NewNode(slice1_out_repr())
                         ->assert_is_op_output("strided_slice", "Out")
                         ->assert_is_op_input("sin", "X");
  slice1->LinksFrom({ew_mul_out}).LinksTo({slice1_out});
  auto* sin = pattern->NewNode(sin_repr())->assert_is_op("sin");
  auto* sin_out = pattern->NewNode(sin_out_repr())
                      ->assert_is_op_output("sin", "Out")
                      ->assert_is_op_nth_input("stack", "X", 0);
  sin->LinksFrom({slice1_out}).LinksTo({sin_out});
  auto* slice2 =
      pattern->NewNode(slice2_repr())
          ->assert_is_op("strided_slice")
          ->assert_more([&](Node* node) {
            auto* op_desc = node->Op();
            return op_desc->GetAttrIfExists<std::vector<int>>("axes") ==
                       std::vector<int>{2} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("starts") ==
                       std::vector<int>{1} &&
                   op_desc->GetAttrIfExists<std::vector<int>>("strides") ==
                       std::vector<int>{2};
          });
  auto* slice2_out = pattern->NewNode(slice2_out_repr())
                         ->assert_is_op_output("strided_slice", "Out")
                         ->assert_is_op_input("cos", "X");
  slice2->LinksFrom({ew_mul_out}).LinksTo({slice2_out});
  auto* cos = pattern->NewNode(cos_repr())->assert_is_op("cos");
  auto* cos_out = pattern->NewNode(cos_out_repr())
                      ->assert_is_op_output("cos", "Out")
                      ->assert_is_op_nth_input("stack", "X", 1);
  cos->LinksFrom({slice2_out}).LinksTo({cos_out});
  auto* stack = pattern->NewNode(stack_repr())
                    ->assert_is_op("stack")
                    ->assert_more([&](Node* node) {
                      auto* op_desc = node->Op();
                      return op_desc->GetAttrIfExists<int>("axis") == 3;
                    });
  auto* stack_out = pattern->NewNode(stack_out_repr())
                        ->assert_is_op_output("stack", "Y")
                        ->assert_is_op_input("flatten_contiguous_range", "X");
  stack->LinksFrom({sin_out, cos_out}).LinksTo({stack_out});

  auto* flatten =
      pattern->NewNode(flatten_repr())
          ->assert_is_op("flatten_contiguous_range")
          ->assert_more([&](Node* node) {
            auto* op_desc = node->Op();
            return op_desc->GetAttrIfExists<int>("start_axis") == 2 &&
                   op_desc->GetAttrIfExists<int>("stop_axis") == 3;
          });
  auto* flatten_out =
      pattern->NewNode(flatten_out_repr())
          ->assert_is_op_output("flatten_contiguous_range", "Out")
          ->AsOutput();
  flatten->LinksFrom({stack_out}).LinksTo({flatten_out});
}

}  // namespace patterns

class SinePosFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"sine_pos_fuse_pass"};
};

void SinePosFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::SinePosXPUPattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle SinePosFusePass fuse";
    /* declare operator node's name */
    // declare operator node's name
    GET_IR_NODE(ew_mul);
    GET_IR_NODE(slice1);
    GET_IR_NODE(slice2);
    GET_IR_NODE(sin);
    GET_IR_NODE(cos);
    GET_IR_NODE(stack);
    GET_IR_NODE(flatten);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(y);
    GET_IR_NODE(ew_mul_out);
    GET_IR_NODE(slice1_out);
    GET_IR_NODE(slice2_out);
    GET_IR_NODE(sin_out);
    GET_IR_NODE(cos_out);
    GET_IR_NODE(stack_out);
    GET_IR_NODE(flatten_out);
    auto* block = flatten->Op()->Block();
    auto* scope = param_scope();
    PADDLE_ENFORCE_NOT_NULL(
        scope, common::errors::InvalidArgument("Scope cannot be nullptr."));
    // Generate sine_pos_xpu fused op
    framework::OpDesc fused_op_desc(block);
    fused_op_desc.SetType("sine_pos_xpu");
    // set attrs for fused op
    fused_op_desc.SetInput("x", {x->Name()});
    fused_op_desc.SetInput("y", {y->Name()});

    fused_op_desc.SetOutput("out", {flatten_out->Name()});
    // relink fused op
    auto* fused_op = graph->CreateOpNode(&fused_op_desc);
    IR_NODE_LINK_TO(x, fused_op);
    IR_NODE_LINK_TO(y, fused_op);
    IR_NODE_LINK_TO(fused_op, flatten_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {ew_mul,
                                                    ew_mul_out,
                                                    slice1,
                                                    slice1_out,
                                                    slice2,
                                                    slice2_out,
                                                    sin,
                                                    sin_out,
                                                    cos,
                                                    cos_out,
                                                    stack,
                                                    stack_out,
                                                    flatten};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);

  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(sine_pos_fuse_pass, paddle::framework::ir::SinePosFusePass);

REGISTER_PASS_CAPABILITY(sine_pos_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "sin_pos_xpu", 0));
