
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

#include "paddle/fluid/framework/ir/transpose_reshape_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

TransposeReshapeFusePass::TransposeReshapeFusePass() {
  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("transpose_reshape_fusion"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("shape")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("trans_first")
      .IsType<bool>()
      .End();
}

int TransposeReshapeFusePass::ApplyTRPattern(Graph* graph) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  auto* x = pattern->NewNode("x")
                ->AsInput()
                ->assert_is_op_input("transpose2", "X")
                ->AsInput();
  auto* transpose =
      pattern->NewNode("transpose_op")->assert_is_op("transpose2");
  auto* transpose_out_var = pattern->NewNode("transpose_op_out_var")
                                ->assert_is_op_output("transpose2", "Out")
                                ->AsIntermediate();
  auto* transpose_xshape_var = pattern->NewNode("transpose_op_xshpae_var")
                                   ->assert_is_op_output("transpose2", "XShape")
                                   ->AsIntermediate();
  auto* reshape = pattern->NewNode("reshape_op")->assert_is_op("reshape2");
  auto* reshape_out_var = pattern->NewNode("reshape_op_out_var")
                              ->assert_is_op_output("reshape2", "Out")
                              ->AsOutput();
  auto* reshape_xshape_var = pattern->NewNode("reshape_op_xshpae_var")
                                 ->assert_is_op_output("reshape2", "XShape")
                                 ->AsIntermediate();

  transpose->LinksFrom({x}).LinksTo({transpose_out_var, transpose_xshape_var});
  reshape->LinksFrom({transpose_out_var})
      .LinksTo({reshape_out_var, reshape_xshape_var});

  int found_fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    auto* transpose_op_node = subgraph.at(transpose);
    auto* reshape_op_node = subgraph.at(reshape);
    // If reshape op has ShapeTensor input or Shape input, we should not fuse.
    if (reshape_op_node->inputs.size() > 1) {
      return;
    }

    // Create an fusion Node.
    OpDesc desc(transpose_op_node->Op()->Block());
    desc.SetType("transpose_reshape_fusion");
    desc.SetInput("X", {subgraph.at(x)->Name()});
    desc.SetOutput("Out", {subgraph.at(reshape_out_var)->Name()});
    desc.SetAttr("trans_first", true);
    desc.SetAttr("axis",
                 PADDLE_GET_CONST(std::vector<int>,
                                  transpose_op_node->Op()->GetAttr("axis")));
    desc.SetAttr("shape",
                 PADDLE_GET_CONST(std::vector<int>,
                                  reshape_op_node->Op()->GetAttr("shape")));
    desc.Flush();

    if (!IsCompat(desc)) {
      LOG(WARNING) << "transpose_reshape fuse pass in out "
                      "transpose_reshape_fusion op compat failed.";
      return;
    }

    auto fusion_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph,
                         {transpose_op_node,
                          reshape_op_node,
                          subgraph.at(transpose_out_var),
                          subgraph.at(transpose_xshape_var),
                          subgraph.at(reshape_xshape_var)});

    IR_NODE_LINK_TO(subgraph.at(x), fusion_node);
    IR_NODE_LINK_TO(fusion_node, subgraph.at(reshape_out_var));

    found_fusion_count++;
  };
  gpd(graph, handler);
  return found_fusion_count;
}

int TransposeReshapeFusePass::ApplyRTPattern(Graph* graph) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  auto* x = pattern->NewNode("x")
                ->AsInput()
                ->assert_is_op_input("reshape2", "X")
                ->AsInput();
  auto* reshape = pattern->NewNode("reshape_op")->assert_is_op("reshape2");
  auto* reshape_out_var = pattern->NewNode("reshape_op_out_var")
                              ->assert_is_op_output("reshape2", "Out")
                              ->AsIntermediate();
  auto* reshape_xshape_var = pattern->NewNode("reshape_op_xshpae_var")
                                 ->assert_is_op_output("reshape2", "XShape")
                                 ->AsIntermediate();
  auto* transpose =
      pattern->NewNode("transpose_op")->assert_is_op("transpose2");
  auto* transpose_out_var = pattern->NewNode("transpose_op_out_var")
                                ->assert_is_op_output("transpose2", "Out")
                                ->AsOutput();
  auto* transpose_xshape_var = pattern->NewNode("transpose_op_xshpae_var")
                                   ->assert_is_op_output("transpose2", "XShape")
                                   ->AsIntermediate();
  reshape->LinksFrom({x}).LinksTo({reshape_out_var, reshape_xshape_var});
  transpose->LinksFrom({reshape_out_var})
      .LinksTo({transpose_out_var, transpose_xshape_var});

  int found_fusion_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    auto* transpose_op_node = subgraph.at(transpose);
    auto* reshape_op_node = subgraph.at(reshape);
    // If reshape op has ShapeTensor input or Shape input, we should not fuse.
    if (reshape_op_node->inputs.size() > 1) {
      return;
    }

    // Create an fusion Node.
    OpDesc desc(transpose_op_node->Op()->Block());
    desc.SetType("transpose_reshape_fusion");
    desc.SetInput("X", {subgraph.at(x)->Name()});
    desc.SetOutput("Out", {subgraph.at(transpose_out_var)->Name()});
    desc.SetAttr("trans_first", false);
    desc.SetAttr("axis",
                 PADDLE_GET_CONST(std::vector<int>,
                                  transpose_op_node->Op()->GetAttr("axis")));
    desc.SetAttr("shape",
                 PADDLE_GET_CONST(std::vector<int>,
                                  reshape_op_node->Op()->GetAttr("shape")));
    desc.Flush();

    if (!IsCompat(desc)) {
      LOG(WARNING) << "transpose_reshape fuse pass in out "
                      "transpose_reshape_fusion op compat failed.";
      return;
    }

    auto fusion_node = g->CreateOpNode(&desc);  // OpDesc will be copied.
    GraphSafeRemoveNodes(graph,
                         {transpose_op_node,
                          reshape_op_node,
                          subgraph.at(reshape_out_var),
                          subgraph.at(transpose_xshape_var),
                          subgraph.at(reshape_xshape_var)});

    IR_NODE_LINK_TO(subgraph.at(x), fusion_node);
    IR_NODE_LINK_TO(fusion_node, subgraph.at(transpose_out_var));

    found_fusion_count++;
  };
  gpd(graph, handler);
  return found_fusion_count;
}

void TransposeReshapeFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("transpose_reshape_fuse", graph);

  int found_fusion_count = ApplyTRPattern(graph);
  int rt_count = ApplyRTPattern(graph);
  AddStatis(found_fusion_count + rt_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(transpose_reshape_fuse_pass,
              paddle::framework::ir::TransposeReshapeFusePass);

REGISTER_PASS_CAPABILITY(transpose_reshape_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("transpose_reshape_fusion", 0));
