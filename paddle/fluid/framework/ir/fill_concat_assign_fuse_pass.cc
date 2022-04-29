// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fill_concat_assign_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/string/pretty_log.h"
namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {
PDNode* FillConcatAssignPattern::operator()() {
  // First path with scale
  auto* fill = pattern->NewNode(fill_repr());
  auto* concat = pattern->NewNode(concat_repr())->assert_is_op("concat");
  auto* assign = pattern->NewNode(assign_repr())->assert_is_op("assign");


  auto* fill_in = pattern->NewNode(fill_in_repr())
                         ->assert_is_op_input("fill_constant_batch_size_like");
  auto* fill_out = pattern->NewNode(fill_out_repr())
                          ->assert_is_op_output("fill_constant_batch_size_like")
                          ->assert_is_op_input("concat");

  auto* concat_in = pattern->NewNode(concat_in_repr())
                           ->assert_is_op_input("concat");

  auto* concat_out = pattern->NewNode(concat_out_repr())
                           ->assert_is_op_output("concat");
  auto* assign_out = pattern->NewNode(assign_out_repr())
                           ->assert_is_op_output("assign");


  fill->LinksFrom({fill_in}).LinksTo({fill_out});
  concat->LinksFrom({fill_out, concat_in})
      .LinksTo({concat_out});
  assign->LinksFrom({concat_out}).LinksTo({assign_out});

  return assign_out;
}
} // namespace patterns

class Node;

FillConcatAssignFusePass::FillConcatAssignFusePass() {
  AddOpCompat(OpCompat("concat"))
      .AddInput("X")  // Input("X"): vector<tensors>
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .End();
}

int FillConcatAssignFusePass::BuildFusion(Graph* graph, const std::string& name_scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  patterns::FillConcatAssignPattern fill_pattern(pattern, name_scope);
  fill_pattern();

  // Create New OpDesc
  auto fuser = [&](Node* assign, Node* concat_in) {
    auto* op_desc = assign->Op();
    op_desc->SetInput("X", {concat_in->Name()});
    IR_NODE_LINK_TO(concat_in, assign);

    return assign;
  };

  int fusion_count=0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    // if (!IsCompat(subgraph, g)) {
    //   LOG(WARNING) << "Pass in op compat failed.";
    //   return;
    // }
    GET_IR_NODE_FROM_SUBGRAPH(fill_in, fill_in, fill_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fill, fill, fill_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fill_out, fill_out, fill_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat_in, concat_in, fill_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat_out, concat_out, fill_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(concat, concat, fill_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(assign, assign, fill_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(assign_out, assign_out, fill_pattern);
    
    fuser(assign, concat_in);
      // Remove unneeded nodes.
    std::unordered_set<const Node*> marked_nodes(
          {fill, fill_out, concat, concat_out});
    GraphSafeRemoveNodes(graph, marked_nodes);
    fusion_count++;
  };

  gpd(graph, handler);

  return fusion_count;
}


void FillConcatAssignFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count = FillConcatAssignFusePass::BuildFusion(graph, name_scope_);

  AddStatis(fusion_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs"))
    string::PrettyLogDetail("---    fused %d pairs of fc gru patterns",
                            fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fill_concat_assign_fuse_pass, paddle::framework::ir::FillConcatAssignFusePass);
REGISTER_PASS_CAPABILITY(fill_concat_assign_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("concat", 0)
            .EQ("assign", 0));
