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
#include "paddle/fluid/framework/ir/sigmoid_elementmul_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir::patterns {

struct SigmoidElementmulFusePattern : public PatternBase {
  SigmoidElementmulFusePattern(PDPattern* pattern,
                               const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(sigmoid);
  PATTERN_DECL_NODE(elementwise_mul);
  // declare variable node's name
  PATTERN_DECL_NODE(sigmoid_x);
  PATTERN_DECL_NODE(sigmoid_out);
  PATTERN_DECL_NODE(elemul_out);
};

SigmoidElementmulFusePattern::SigmoidElementmulFusePattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* sigmoid_x = pattern->NewNode(sigmoid_x_repr())
                        ->assert_is_op_input("sigmoid", "X")
                        ->assert_var_not_persistable();

  auto* sigmoid_op = pattern->NewNode(sigmoid_repr())->assert_is_op("sigmoid");

  auto* sigmoid_out = pattern->NewNode(sigmoid_out_repr())
                          ->assert_is_op_output("sigmoid", "Out")
                          ->assert_var_not_persistable();

  auto* elemul_op =
      pattern->NewNode(elementwise_mul_repr())->assert_is_op("elementwise_mul");

  auto* elemul_out = pattern->NewNode(elemul_out_repr())
                         ->assert_is_op_output("elementwise_mul", "Out")
                         ->assert_var_not_persistable();

  sigmoid_op->LinksFrom({sigmoid_x}).LinksTo({sigmoid_out});
  elemul_op->LinksFrom({sigmoid_x, sigmoid_out}).LinksTo({elemul_out});
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

SigmoidElementmulFusePass::SigmoidElementmulFusePass() = default;

void SigmoidElementmulFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::SigmoidElementmulFusePattern pattern(gpd.mutable_pattern(),
                                                 name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle SigmoidElementmulFusePass fuse";
#define GET_IR_NODE(node_) GET_IR_NODE_FROM_SUBGRAPH(node_, node_, pattern)
    GET_IR_NODE(sigmoid_x);
    GET_IR_NODE(sigmoid);
    GET_IR_NODE(sigmoid_out);
    GET_IR_NODE(elementwise_mul);
    GET_IR_NODE(elemul_out);
#undef GET_IR_NODE
    auto* block = sigmoid->Op()->Block();
    std::string elemul_out_name = elemul_out->Name();

    // Generate swish op
    framework::OpDesc swish_op_desc(block);
    swish_op_desc.SetType("swish");
    swish_op_desc.SetInput("X", {sigmoid_x->Name()});
    swish_op_desc.SetAttr("beta", 1.f);
    swish_op_desc.SetOutput("Out", {elemul_out_name});

    auto* swish = graph->CreateOpNode(&swish_op_desc);
    IR_NODE_LINK_TO(sigmoid_x, swish);
    IR_NODE_LINK_TO(swish, elemul_out);

    // delete useless node
    std::unordered_set<const Node*> delete_nodes;
    delete_nodes = {sigmoid, sigmoid_out, elementwise_mul};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(sigmoid_elementmul_fuse_pass,
              paddle::framework::ir::SigmoidElementmulFusePass);

REGISTER_PASS_CAPABILITY(sigmoid_elementmul_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "swish", 0));
