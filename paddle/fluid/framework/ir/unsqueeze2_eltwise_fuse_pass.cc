/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/unsqueeze2_eltwise_fuse_pass.h"

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct UnsqueezeEltwise : public PatternBase {
  UnsqueezeEltwise(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "unsqueeze2_eltwise_fuse_pass") {}

  PDNode *operator()(PDNode *x, PDNode *y);

  // declare operator node's name
  PATTERN_DECL_NODE(unsqz);
  PATTERN_DECL_NODE(elementwise);
  // declare variable node's name
  PATTERN_DECL_NODE(eltwise_in_x);
  PATTERN_DECL_NODE(unsqz_in);
  PATTERN_DECL_NODE(unsqz_out);
  PATTERN_DECL_NODE(eltwise_out);
};

PDNode *UnsqueezeEltwise::operator()(PDNode *x, PDNode *y) {
  x->assert_is_op_input("elementwise_mul", "X");
  y->assert_is_op_input("unsqueeze2", "X");

  auto *unsqz = pattern->NewNode(unsqz_repr())->assert_is_op("unsqueeze2");
  auto *unsqz_out = pattern->NewNode(unsqz_out_repr())
                        ->assert_is_op_output("unsqueeze2", "Out")
                        ->assert_is_op_input("elementwise_mul", "Y");
  unsqz->LinksFrom({y}).LinksTo({unsqz_out});

  auto *elementwise =
      pattern->NewNode(elementwise_repr())->assert_is_op("elementwise_mul");
  auto *eltwise_out = pattern->NewNode(eltwise_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("elementwise_mul");

  elementwise->LinksFrom({x, unsqz_out}).LinksTo({eltwise_out});
  return eltwise_out;
}

}  // namespace patterns

UnsqueezeEltwiseFusePass::UnsqueezeEltwiseFusePass() {
  AddOpCompat(OpCompat("unsqueeze2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("AxesTensor")
      .IsOptional()
      .IsTensor()
      .End()
      .AddInput("AxesTensorList")
      .IsOptional()
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axes")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("elementwise_mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      // The attribute value is - 1 before fusion and 0 after fusion
      .AddAttr("axis")
      .IsIntIn({-1, 0})
      .End();
}

void UnsqueezeEltwiseFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("unsqueeze2_eltwise_fuse_pass", graph);
  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("unsqueeze2_eltwise_fuse_pass/x")
                ->AsInput()
                ->assert_is_op_input("elementwise_mul", "X")
                ->assert_var_not_persistable();
  auto *y = gpd.mutable_pattern()
                ->NewNode("unsqueeze2_eltwise_fuse_pass/y")
                ->AsInput()
                ->assert_is_op_input("unsqueeze2", "X")
                ->assert_var_not_persistable();
  patterns::UnsqueezeEltwise fused_pattern(gpd.mutable_pattern(),
                                           "unsqueeze2_eltwise_fuse_pass");
  fused_pattern(x, y);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    VLOG(4) << "handle UnsqueezeEltwise fuse";
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_op, elementwise, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_out, eltwise_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(unsqz_op, unsqz, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(unsqz_out, unsqz_out, fused_pattern);

    size_t eltwise_in_x_rank = (subgraph.at(x)->Var()->GetShape()).size();
    size_t unsqz_in_rank = (subgraph.at(y)->Var()->GetShape()).size();
    std::vector<int> unsqz_op_axes =
        PADDLE_GET_CONST(std::vector<int>, unsqz_op->Op()->GetAttr("axes"));
    int eltwise_op_axis =
        PADDLE_GET_CONST(int, eltwise_op->Op()->GetAttr("axis"));

    if (eltwise_in_x_rank == 4 && unsqz_in_rank == 2 &&
        unsqz_op_axes == std::vector<int>{2, 3} && eltwise_op_axis == -1) {
      eltwise_op->Op()->SetAttr("axis", 0);
      eltwise_op->Op()->SetInput("Y", {subgraph.at(y)->Name()});
      IR_NODE_LINK_TO(subgraph.at(x), eltwise_op);
      IR_NODE_LINK_TO(subgraph.at(y), eltwise_op);
      IR_NODE_LINK_TO(eltwise_op, eltwise_out);
      GraphSafeRemoveNodes(graph, {unsqz_op, unsqz_out});
      found_subgraph_count++;
      if (!IsCompat(*eltwise_op->Op())) {
        LOG(WARNING) << "unsqueeze2_eltwise_fuse_pass op compat failed.";
        return;
      }
    }
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(unsqueeze2_eltwise_fuse_pass,
              paddle::framework::ir::UnsqueezeEltwiseFusePass);
REGISTER_PASS_CAPABILITY(unsqueeze2_eltwise_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("unsqueeze2", 0)
            .LE("elementwise_mul", 1));
