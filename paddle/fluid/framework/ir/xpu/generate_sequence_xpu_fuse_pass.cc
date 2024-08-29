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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
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

struct GenerateSequenceXPUPattern : public PatternBase {
  GenerateSequenceXPUPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fill_any_like);
  PATTERN_DECL_NODE(cumsum);
  PATTERN_DECL_NODE(elementwise_sub);
  // declare variable node's name
  PATTERN_DECL_NODE(fill_any_like_x);
  PATTERN_DECL_NODE(fill_any_like_out);
  PATTERN_DECL_NODE(cumsum_out);
  PATTERN_DECL_NODE(elementwise_sub_out);
};

GenerateSequenceXPUPattern::GenerateSequenceXPUPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* fill_any_like_x = pattern->NewNode(fill_any_like_x_repr())
                              ->assert_is_op_input("fill_any_like", "X")
                              ->assert_var_not_persistable()
                              ->assert_more([](Node* node) {
                                return node->Var()->GetShape().size() == 2;
                              });
  auto* fill_any_like =
      pattern->NewNode(fill_any_like_repr())
          ->assert_is_op("fill_any_like")
          ->assert_more([](Node* node) {
            float value = PADDLE_GET_CONST(float, node->Op()->GetAttr("value"));
            return static_cast<int>(value) == 1;
          });
  auto* fill_any_like_out = pattern->NewNode(fill_any_like_out_repr())
                                ->assert_is_op_output("fill_any_like", "Out")
                                ->assert_is_op_input("cumsum", "X")
                                ->assert_is_op_input("elementwise_sub", "Y")
                                ->assert_var_not_persistable()
                                ->assert_has_n_outputs(2);
  auto* cumsum =
      pattern->NewNode(cumsum_repr())
          ->assert_is_op("cumsum")
          ->assert_more([](Node* node) {
            return !PADDLE_GET_CONST(bool, node->Op()->GetAttr("exclusive")) &&
                   !PADDLE_GET_CONST(bool, node->Op()->GetAttr("reverse")) &&
                   !PADDLE_GET_CONST(bool, node->Op()->GetAttr("flatten")) &&
                   ((PADDLE_GET_CONST(int, node->Op()->GetAttr("axis")) == 1) ||
                    (PADDLE_GET_CONST(int, node->Op()->GetAttr("axis")) == -1));
          });
  auto* cumsum_out = pattern->NewNode(cumsum_out_repr())
                         ->assert_is_op_output("cumsum", "Out")
                         ->assert_is_op_input("elementwise_sub", "X")
                         ->assert_var_not_persistable()
                         ->assert_has_n_outputs(1);
  auto* elementwise_sub =
      pattern->NewNode(elementwise_sub_repr())
          ->assert_is_op("elementwise_sub")
          ->assert_more([](Node* node) {
            return PADDLE_GET_CONST(int, node->Op()->GetAttr("axis")) == -1;
          });
  auto* elementwise_sub_out =
      pattern->NewNode(elementwise_sub_out_repr())
          ->assert_is_op_output("elementwise_sub", "Out")
          ->assert_var_not_persistable();
  fill_any_like->LinksFrom({fill_any_like_x}).LinksTo({fill_any_like_out});
  cumsum->LinksFrom({fill_any_like_out}).LinksTo({cumsum_out});
  elementwise_sub->LinksFrom({cumsum_out, fill_any_like_out})
      .LinksTo({elementwise_sub_out});
}

}  // namespace patterns

/*
Origin subgraph:
          fill_any_like
            /     \
            |     |
            |   cumsum
            |     |
            \     /
         elementwise_sub

Fused subgraph:
      generate_sequence_xpu
*/
class GenerateSequenceXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"generate_sequence_xpu_fuse_pass"};
};

void GenerateSequenceXPUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::GenerateSequenceXPUPattern pattern(gpd.mutable_pattern(),
                                               name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle GenerateSequenceXPUFusePass fuse";
    GET_IR_NODE(fill_any_like);
    GET_IR_NODE(cumsum);
    GET_IR_NODE(elementwise_sub);
    GET_IR_NODE(fill_any_like_x);
    GET_IR_NODE(fill_any_like_out);
    GET_IR_NODE(cumsum_out);
    GET_IR_NODE(elementwise_sub_out);

    auto* block = fill_any_like->Op()->Block();
    framework::OpDesc op_desc(block);
    op_desc.SetType("generate_sequence_xpu");
    op_desc.SetInput("x", {fill_any_like_x->Name()});
    op_desc.SetOutput("out", {elementwise_sub_out->Name()});
    op_desc.SetAttr(
        "dtype", PADDLE_GET_CONST(int, fill_any_like->Op()->GetAttr("dtype")));
    auto* generate_sequence_xpu = graph->CreateOpNode(&op_desc);
    IR_NODE_LINK_TO(fill_any_like, generate_sequence_xpu);
    IR_NODE_LINK_TO(generate_sequence_xpu, elementwise_sub_out);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes{
        fill_any_like, fill_any_like_out, cumsum, cumsum_out, elementwise_sub};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(generate_sequence_xpu_fuse_pass,
              paddle::framework::ir::GenerateSequenceXPUFusePass);

REGISTER_PASS_CAPABILITY(generate_sequence_xpu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "generate_sequence_xpu", 0));
