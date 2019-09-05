// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/remove_reshape_transpose_pass.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

static int ReshapeTranspose(Graph* graph) {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("rm_head/x")
                ->AsInput()
                ->assert_is_op_input("reshape2", "X");
  patterns::RmReshapeTranspose rm_reshape_transpose_pattern(
      gpd.mutable_pattern(), "remove_reshape_transpose");
  rm_reshape_transpose_pattern(x, false /*inverse*/, false);
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle remove reshape transpose pattern";
    GET_IR_NODE_FROM_SUBGRAPH(reshape, reshape2, rm_reshape_transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose, transpose2,
                              rm_reshape_transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul, matmul, rm_reshape_transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape2_out,
                              rm_reshape_transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose2_out,
                              rm_reshape_transpose_pattern);

    auto reshape_desc = reshape->Op();
    int head_number =
        boost::get<std::vector<int>>(reshape_desc->GetAttr("shape")).at(2);
    auto matmul_op_desc = matmul->Op();
    matmul_op_desc->SetAttr("head_number", head_number);
    matmul_op_desc->RenameInput(transpose_out->Name(), subgraph.at(x)->Name());

    PADDLE_ENFORCE_NE(subgraph.count(x), 0);
    IR_NODE_LINK_TO(subgraph.at(x), matmul);
    GraphSafeRemoveNodes(graph,
                         {reshape, transpose, reshape_out, transpose_out});
    found_count++;
  };
  gpd(graph, handler);
  return found_count;
}

static int TransposeReshape(Graph* graph) {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("rm_inverse/x")
                ->AsOutput()
                ->assert_is_op_output("matmul", "Out");
  patterns::RmReshapeTranspose rm_transpose_reshape_pattern(
      gpd.mutable_pattern(), "remove_transpose_reshape");
  rm_transpose_reshape_pattern(x, true, false);
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle remove transpose reshape pattern";
    GET_IR_NODE_FROM_SUBGRAPH(reshape, reshape2, rm_transpose_reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose, transpose2,
                              rm_transpose_reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape2_out,
                              rm_transpose_reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose2_out,
                              rm_transpose_reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out,
                              rm_transpose_reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul, matmul, rm_transpose_reshape_pattern);

    auto matmul_op_desc = matmul->Op();
    matmul_op_desc->RenameOutput(matmul_out->Name(), reshape_out->Name());
    IR_NODE_LINK_TO(matmul, reshape_out);
    GraphSafeRemoveNodes(graph,
                         {reshape, transpose, matmul_out, transpose_out});
    found_count++;
  };
  gpd(graph, handler);
  return found_count;
}

static int ReshapeTransposeScale(Graph* graph) {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("rm_head_scale/x")
                ->AsInput()
                ->assert_is_op_input("reshape2", "X");
  patterns::RmReshapeTranspose rm_reshape_transpose_scale_pattern(
      gpd.mutable_pattern(), "remove_reshape_transpose_scale");
  rm_reshape_transpose_scale_pattern(x, false /*inverse*/, true);
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle remove reshape transpose scale pattern";
    GET_IR_NODE_FROM_SUBGRAPH(reshape, reshape2,
                              rm_reshape_transpose_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose, transpose2,
                              rm_reshape_transpose_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape2_out,
                              rm_reshape_transpose_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose2_out,
                              rm_reshape_transpose_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul, matmul,
                              rm_reshape_transpose_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, rm_reshape_transpose_scale_pattern);

    auto reshape_desc = reshape->Op();
    int head_number =
        boost::get<std::vector<int>>(reshape_desc->GetAttr("shape")).at(2);
    auto matmul_op_desc = matmul->Op();
    matmul_op_desc->SetAttr("head_number", head_number);
    auto scale_op_desc = scale->Op();
    scale_op_desc->RenameInput(transpose_out->Name(), subgraph.at(x)->Name());

    PADDLE_ENFORCE_NE(subgraph.count(x), 0);
    IR_NODE_LINK_TO(subgraph.at(x), scale);
    GraphSafeRemoveNodes(graph,
                         {reshape, transpose, reshape_out, transpose_out});
    found_count++;
  };
  gpd(graph, handler);
  return found_count;
}

static int RemoveStack(Graph* graph) {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("rm_stack/x")
                ->AsInput()
                ->assert_is_op_input("stack", "X");
  patterns::RemoveStack remove_stack_pattern(gpd.mutable_pattern(),
                                             "remove_stack_pattern");
  remove_stack_pattern(x);
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle remove stack pattern";
    GET_IR_NODE_FROM_SUBGRAPH(stack, stack, remove_stack_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(stack_out, stack_out, remove_stack_pattern);

    auto stack_op_desc = stack->Op();
    stack_op_desc->SetType("concat");
    stack_op_desc->SetAttr("axis", 2);
    stack_op_desc->SetOutput("Out",
                             std::vector<std::string>({stack_out->Name()}));

    found_count++;
  };
  gpd(graph, handler);
  return found_count;
}

static int DetectStack(Graph* graph) {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("rm_stack/x")
                ->AsInput()
                ->assert_is_op_input("stack", "X");
  patterns::RemoveStack remove_stack_pattern(gpd.mutable_pattern(),
                                             "remove_stack_pattern");
  remove_stack_pattern(x);
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle remove stack pattern";
    GET_IR_NODE_FROM_SUBGRAPH(stack, stack, remove_stack_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(stack_out, stack_out, remove_stack_pattern);

    found_count++;
  };
  gpd(graph, handler);
  return found_count;
}

void RemoveReshapeTransposeForAttentionPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph);
  FusePassBase::Init("remove_reshape_transpose", graph);
  int count0 = 0, count1 = 0, count2 = 0, count3 = 0;
  count0 = DetectStack(graph);
  if (count0 != 0) {
    count1 = ReshapeTranspose(graph);
    count2 = TransposeReshape(graph);
    count3 = ReshapeTransposeScale(graph);
    if (count1 && count2 && count3) {
      RemoveStack(graph);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(remove_reshape_transpose_pass,
              paddle::framework::ir::RemoveReshapeTransposeForAttentionPass);
