// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/reverse_roll_fuse_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#define GET_IR_NODE(node__) \
  GET_IR_NODE_FROM_SUBGRAPH(node__, node__, reverse_roll_pattern);
#define GET_NODES                 \
  GET_IR_NODE(reshape2_00_op);    \
  GET_IR_NODE(reshape2_00_out);   \
  GET_IR_NODE(reshape2_10_op);    \
  GET_IR_NODE(reshape2_10_out);   \
  GET_IR_NODE(transpose2_20_op);  \
  GET_IR_NODE(transpose2_20_out); \
  GET_IR_NODE(reshape2_30_op);    \
  GET_IR_NODE(reshape2_30_out);   \
  GET_IR_NODE(reshape2_50_op);    \
  GET_IR_NODE(reshaep2_50_out);

namespace paddle {
namespace framework {
namespace ir {
class Node;
ReverseRollFusePass::ReverseRollFusePass() {
  AddOpCompat(OpCompat("reshape2"))
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
  AddOpCompat(OpCompat("roll"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsType<std::vector<int64_t>>()
      .End()
      .AddAttr("shifts")
      .IsType<std::vector<int64_t>>()
      .End();
}
int ReverseRollFusePass::ApplyPattern(ir::Graph* graph, bool with_roll) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      platform::errors::InvalidArgument(
          "The input graph of ReverseRollFusePass should not be "
          "nullptr."));
  GraphPatternDetector gpd;
  FusePassBase::Init(scope_name_, graph);
  PDNode* x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_op_input("reshape2", "X")
                  ->AsInput();
  patterns::ReverseRollPattern reverse_roll_pattern(
      gpd.mutable_pattern(), scope_name_, with_roll);
  reverse_roll_pattern(x);
  int fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "reverse roll in op compat failed.";
      return;
    }
    if (with_roll) {
      VLOG(4) << "reverse_roll_fuse pass, shift_size>0, with roll op";
    } else {
      VLOG(4) << "reverse_roll_fuse pass, shift_size=0, without roll op";
    }
    GET_NODES;
    Node* roll_40_op = nullptr;
    Node* roll_40_out = nullptr;
    if (with_roll) {
      GET_IR_NODE_FROM_SUBGRAPH(
          tmp_roll_40_op, roll_40_op, reverse_roll_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(
          tmp_roll_40_out, roll_40_out, reverse_roll_pattern);
      roll_40_op = tmp_roll_40_op;
      roll_40_out = tmp_roll_40_out;
    }

    std::unordered_set<const Node*> del_node_set = {reshape2_00_op,
                                                    reshape2_00_out,
                                                    reshape2_10_op,
                                                    reshape2_10_out,
                                                    transpose2_20_op,
                                                    transpose2_20_out,
                                                    reshape2_30_op,
                                                    reshape2_30_out,
                                                    reshape2_50_op};
    if (with_roll) {
      del_node_set.insert(roll_40_op);
      del_node_set.insert(roll_40_out);
    }

    std::vector<int32_t> reshape2_10_attr_shape = PADDLE_GET_CONST(
        std::vector<int32_t>, reshape2_10_op->Op()->GetAttr("shape"));
    if (reshape2_10_attr_shape[1] <= 0) {
      return;
    }
    if (reshape2_10_attr_shape[1] != reshape2_10_attr_shape[2]) {
      return;
    }
    int window_number = reshape2_10_attr_shape[1] * reshape2_10_attr_shape[2];
    std::vector<int> reshape_2_00_attr_shape = PADDLE_GET_CONST(
        std::vector<int>, reshape2_00_op->Op()->GetAttr("shape"));
    int window_size_h = reshape_2_00_attr_shape[1];
    if (window_size_h <= 0) {
      return;
    }
    int window_size_w = reshape_2_00_attr_shape[2];
    if (window_size_h != window_size_w) {
      return;
    }
    int window_size = window_size_h;
    int window_len = window_size_h * window_size_w;
    int input_resolution = reshape2_10_attr_shape[1] * window_size_h;

    auto shift_size = 0;
    if (with_roll) {
      std::vector<int64_t> roll_40_op_attr_shifts = PADDLE_GET_CONST(
          std::vector<int64_t>, roll_40_op->Op()->GetAttr("shifts"));
      if (roll_40_op_attr_shifts[0] != roll_40_op_attr_shifts[1]) {
        return;
      }
      shift_size = roll_40_op_attr_shifts[0];
    }
    OpDesc reverse_roll_desc(reshape2_00_op->Op()->Block());
    reverse_roll_desc.SetType("reverse_roll");
    reverse_roll_desc.SetInput("X", {subgraph.at(x)->Name()});
    reverse_roll_desc.SetOutput("Out", {reshaep2_50_out->Name()});
    reverse_roll_desc.SetAttr("window_number", window_number);
    reverse_roll_desc.SetAttr("window_size", window_size);
    reverse_roll_desc.SetAttr("window_len", window_len);
    reverse_roll_desc.SetAttr("shift_size", static_cast<int>(shift_size));
    reverse_roll_desc.SetAttr("input_resolution", input_resolution);
    auto reverse_roll_node = graph->CreateOpNode(&reverse_roll_desc);
    IR_NODE_LINK_TO(subgraph.at(x), reverse_roll_node);
    IR_NODE_LINK_TO(reverse_roll_node, reshaep2_50_out);
    GraphSafeRemoveNodes(graph, del_node_set);
    ++fuse_count;
  };
  gpd(graph, handler);
  return fuse_count;
}
void ReverseRollFusePass::ApplyImpl(ir::Graph* graph) const {
  int fuse_count = 0;
  fuse_count += ApplyPattern(graph, true);
  fuse_count += ApplyPattern(graph, false);
  AddStatis(fuse_count);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reverse_roll_fuse_pass,
              paddle::framework::ir::ReverseRollFusePass);
REGISTER_PASS_CAPABILITY(reverse_roll_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("transpose2", 0)
            .EQ("reshape2", 0));
