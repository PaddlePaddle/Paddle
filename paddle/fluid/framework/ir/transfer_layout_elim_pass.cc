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

#include "paddle/fluid/framework/ir/transfer_layout_elim_pass.h"
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

// (D) means deleted nodes
// (G) means generated node
//        var0               var0'                var0               var0'
//         |                  |                    |                   |
// transfer_layout0(D)  transfer_layout0'(D)       |                   |
//         |                  |                    |                   |
//       var1(D)           var1'(D)  ->            |                   |
//         \                 /                      \                 /
//               op_node                 ->                op_node
//                 |                                          |
//                 |                                        var2
//                 |                                          |
//                 |                                    transfer_layout(G)
//                 |                                          |
//                var2                              var2'(var2 + suffix)(G)
//                 |                                           |
//              other ops                                  other ops
// Put transfer_layout after op_node
void TransferLayoutElimPass::PutTranferlayoutAfterOp(Node *op_node,
                                                     ir::Graph *graph) const {
  std::unordered_set<const Node *> remove_nodes;
  // Ensure op_node has only one output!
  int op_node_useful_output = 0;
  Node *var2;
  for (auto ele : op_node->outputs) {
    if (ele->outputs.size() >= 1) {
      op_node_useful_output++;
      var2 = ele;
    }
  }
  CHECK_EQ(op_node_useful_output == 1, true);

  // group_norm has 3 inputs, but we do not need there is a transfer_layout
  // before Bias and Scale so we extract useful_var1s from op_node->inputs.
  std::vector<Node *> useful_var1s;
  for (auto var1 : op_node->inputs) {
    // if (var1->inputs.size() >= 1 &&
    //         var1->inputs[0]->Op()->Type() == "transfer_layout") {
    //   useful_var1s.push_back(var1);
    // }
    useful_var1s.push_back(var1);
  }
  CHECK_EQ(useful_var1s.size() >= 1L, true);

  auto transfer_layout_opdesc = *useful_var1s[0]->inputs[0]->Op()->Proto();
  auto block = useful_var1s[0]->inputs[0]->Op()->Block();

  framework::OpDesc new_transfer_layout_desc(transfer_layout_opdesc, block);
  new_transfer_layout_desc.SetInput("X", {var2->Name()});

  // Do not use this line code, may result in failing SetShape in netron
  // display.
  // auto *var2_desc = block->Var(var2->Name());
  auto *var2_desc = var2->Var();
  auto var2_shape = var2_desc->GetShape();
  CHECK_EQ(var2_shape.size() >= 4L, true);
  auto new_var2_shape = var2_shape;

  std::string suffix = "_nchw_to_nhwc";
  auto dst_layout = static_cast<DataLayout>(
      new_transfer_layout_desc.GetAttrIfExists<int>("dst_layout"));
  auto src_layout = static_cast<DataLayout>(
      new_transfer_layout_desc.GetAttrIfExists<int>("src_layout"));
  if (dst_layout == DataLayout::NCHW && src_layout == DataLayout::NHWC) {
    suffix = "_nhwc_to_nchw";
    new_var2_shape[1] = var2_shape[2];
    new_var2_shape[2] = var2_shape[3];
    new_var2_shape[3] = var2_shape[1];
  } else if (dst_layout == DataLayout::NHWC && src_layout == DataLayout::NCHW) {
    suffix = "_nchw_to_nhwc";
    new_var2_shape[1] = var2_shape[3];
    new_var2_shape[2] = var2_shape[1];
    new_var2_shape[3] = var2_shape[2];
  }

  var2_desc->SetShape(new_var2_shape);

  std::string var2_dot_name = var2->Name() + suffix;
  new_transfer_layout_desc.SetOutput("Out", {var2_dot_name});
  new_transfer_layout_desc.Flush();

  auto *var2_dot_desc = block->Var(var2_dot_name);
  var2_dot_desc->SetPersistable(false);
  // set var2_dot_desc be var2_shape
  var2_dot_desc->SetShape(var2_shape);

  var2_dot_desc->SetDataType(var2->Var()->GetDataType());
  auto var2_dot = graph->CreateVarNode(var2_dot_desc);
  auto *new_transfer_layout_node =
      graph->CreateOpNode(&new_transfer_layout_desc);

  for (auto other_op : var2->outputs) {
    IR_NODE_UNLINK(var2, other_op);
    other_op->Op()->RenameInput(var2->Name(), var2_dot_name);
    IR_NODE_LINK_TO(var2_dot, other_op);
  }

  IR_NODE_LINK_TO(var2, new_transfer_layout_node);
  IR_NODE_LINK_TO(new_transfer_layout_node, var2_dot);

  for (auto var1 : useful_var1s) {
    auto transfer_layout0_op = var1->inputs[0];
    auto var0 = transfer_layout0_op->inputs[0];
    IR_NODE_UNLINK(var0, transfer_layout0_op);
    // IR_NODE_UNLINK(var1, op_node);
    IR_NODE_LINK_TO(var0, op_node);

    op_node->Op()->RenameInput(var1->Name(), var0->Name());
    remove_nodes.emplace(transfer_layout0_op);
    remove_nodes.emplace(var1);
  }

  GraphSafeRemoveNodes(graph, remove_nodes);
}

bool TransferLayoutElimPass::AllInputIsTransferlayout(
    const ir::Node *op_node) const {
  std::set<int> dst_layouts;
  std::set<int> src_layouts;

  auto *scope = param_scope();

  for (auto var : op_node->inputs) {
    // If this input is a 1D persistable tensor，we allow transfer_layout not
    // appear before this var, but temporarily diasble this if.
    if (var->Var()->Persistable() && 0) {
      auto var_dims =
          scope->FindVar(var->Name())->GetMutable<phi::DenseTensor>()->dims();
      if (var_dims.size() == 1) {
        continue;
      }
    }

    if (var->inputs.size() != 1L) {
      return false;
    }
    if (var->outputs.size() != 1L) {
      return false;
    }
    if (var->inputs[0]->Name() != "transfer_layout") {
      return false;
    }
    auto transfer_layout_desc = var->inputs[0]->Op();
    dst_layouts.insert(
        transfer_layout_desc->GetAttrIfExists<int>("dst_layout"));
    src_layouts.insert(
        transfer_layout_desc->GetAttrIfExists<int>("src_layout"));
  }

  // Make sure the dst_layout and src_layout attribute is same so that these
  // transfer_layout can be moved down.
  return dst_layouts.size() == 1 && src_layouts.size() == 1;
}

// (D) means deleted nodes
// (G) means generated node
//            var0
//              |
//      transfer_layout0(D)
//              |
//            var1
//              |
//      transfer_layout1(D ,op_node)
//              |
//             var2
//         |   |     |
//       op0   op1    op2

void TransferLayoutElimPass::ElimTwoTranferlayout(Node *op_node,
                                                  ir::Graph *graph,
                                                  bool *modify) const {
  std::unordered_set<const Node *> remove_nodes;
  auto var1 = op_node->inputs[0];
  auto transfer_layout0 = var1->inputs[0];
  auto var0 = transfer_layout0->inputs[0];
  auto var2 = op_node->outputs[0];
  CHECK_EQ(transfer_layout0->Name() == "transfer_layout", true);
  CHECK_EQ(op_node->Name() == "transfer_layout", true);
  int dst0 = transfer_layout0->Op()->GetAttrIfExists<int>("dst_layout");
  int src0 = transfer_layout0->Op()->GetAttrIfExists<int>("src_layout");
  int dst1 = op_node->Op()->GetAttrIfExists<int>("dst_layout");
  int src1 = op_node->Op()->GetAttrIfExists<int>("src_layout");

  if (!(dst0 == src1 && dst1 == src0)) {
    // We can not eliminate these two transfer_layout.
    *modify = false;
    return;
  }

  *modify = true;
  remove_nodes.emplace(transfer_layout0);
  remove_nodes.emplace(var1);
  remove_nodes.emplace(op_node);
  remove_nodes.emplace(var2);

  for (auto next_op : var2->outputs) {
    IR_NODE_LINK_TO(var0, next_op);
    next_op->Op()->RenameInput(var2->Name(), var0->Name());
  }
  GraphSafeRemoveNodes(graph, remove_nodes);
}

void TransferLayoutElimPass::ApplyImpl(ir::Graph *graph) const {
  const std::string pattern_name = "transfer_layout_elim_pass";
  FusePassBase::Init(pattern_name, graph);

  auto transfer_format = [&](std::string data_format) -> std::string {
    if (data_format == "NCHW") {
      return "NHWC";
    } else if (data_format == "NHWC") {
      return "NCHW";
    }
    return "";
  };

  while (true) {
    auto op_node_sorted = framework::ir::TopologyVarientSort(
        *graph, static_cast<framework::ir::SortKind>(0));
    bool modify = false;
    for (auto *op_node : op_node_sorted) {
      if (!op_node->IsOp()) continue;

      // For these Ops, you can move down the transfer_layout without changing
      // any attribute!
      std::vector<std::string> act_like_ops = {
          "elementwise_add",
          "hard_swish",
          "silu",
      };
      bool is_act_like_op =
          find(act_like_ops.begin(), act_like_ops.end(), op_node->Name()) !=
          act_like_ops.end();
      // For these Ops, you can move down the transfer_layout, but MUST change
      // the data_format attribute!
      std::vector<std::string> pool_like_ops = {
          // "pool2d",
          // "group_norm",
      };
      bool is_pool_like_op =
          find(pool_like_ops.begin(), pool_like_ops.end(), op_node->Name()) !=
          pool_like_ops.end();
      // For these Ops, you can move down the transfer_layout, but MUST change
      // the axis attribute!
      std::vector<std::string> concat_like_ops = {
          "concat",
      };
      bool is_concat_like_op = find(concat_like_ops.begin(),
                                    concat_like_ops.end(),
                                    op_node->Name()) != concat_like_ops.end();
      bool is_elim_op = op_node->Name() == "transfer_layout";

      if (!(is_act_like_op || is_concat_like_op || is_pool_like_op ||
            is_elim_op))
        continue;

      if (AllInputIsTransferlayout(op_node)) {
        if (is_concat_like_op) {
          PutTranferlayoutAfterOp(op_node, graph);
          int axis = op_node->Op()->GetAttrIfExists<int>("axis");
          if (axis == 1) {
            op_node->Op()->SetAttr("axis", 3);
          } else if (axis == 2) {
            op_node->Op()->SetAttr("axis", 1);
          } else if (axis == 3) {
            op_node->Op()->SetAttr("axis", 2);
          }
          modify = true;
          break;
        }
        if (is_pool_like_op) {
          PutTranferlayoutAfterOp(op_node, graph);
          op_node->Op()->SetAttr(
              "data_format",
              transfer_format(
                  op_node->Op()->GetAttrIfExists<std::string>("data_format")));
          modify = true;
          break;
        }
        if (is_act_like_op) {
          PutTranferlayoutAfterOp(op_node, graph);
          modify = true;
          break;
        }
        if (is_elim_op) {
          ElimTwoTranferlayout(op_node, graph, &modify);
          break;
        }
      }
    }
    if (!modify) break;
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(transfer_layout_elim_pass,
              paddle::framework::ir::TransferLayoutElimPass);
// Add below for test_transfer_elim_pass passing.
REGISTER_PASS_CAPABILITY(transfer_layout_elim_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination());
