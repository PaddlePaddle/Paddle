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

#include "paddle/fluid/framework/ir/transfer_layout_elim.h"

#include <string>
#include "paddle/fluid/framework/op_version_registry.h"

#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

// (X) means deleted nodes
// (G) means generated node
//        var0-----------------var0'                var0-----------------var0'
//         |                  |                       |                   |
// transfer_layout0(X)  transfer_layout0'(X)          |                   |
//         |                  |                       |                   |
//       var1(X)-----------------var1'(X)  ->         |                   |
//         |                  |                       |                   |  
//               op_node                                      op_node
//                 |                                             |
//                 |                                           var2'(G)
//                 |                                             |
//                 |                                       transfer_layout(G) 
//                 |                                             |
//                var2                                          var2 
//                 |                                              |
//              other ops                                     other ops

void TransferLayoutElimPass::PutTranferlayoutAfterOp(Node *op_node, ir::Graph* graph) const {
  std::unordered_set<const Node *> remove_nodes;
  CHECK_EQ(op_node->outputs.size() == 1L, true);
  auto var2 = op_node->outputs[0];
  auto transfer_layout_opdesc = *op_node->inputs[0]->inputs[0]->Op()->Proto();
  auto block = op_node->inputs[0]->inputs[0]->Op()->Block();;
  framework::OpDesc new_transfer_layout_desc(transfer_layout_opdesc, block);
  new_transfer_layout_desc.SetOutput("Out", {var2->Name()});
  
  std::string name = "哈哈";
  for (auto var1 : op_node->inputs) {
    auto transfer_layout0_op = var1->inputs[0];
    auto var0 = transfer_layout0_op->inputs[0];
    name += var0->Name();
  }
  new_transfer_layout_desc.SetInput("X", {name});
  new_transfer_layout_desc.Flush();

  auto *var2_dot_desc = block->Var(name);
  var2_dot_desc->SetPersistable(false);
  var2_dot_desc->SetShape({-1, -1, -1, -1});
  var2_dot_desc->SetDataType(var2->Var()->GetDataType());
  auto var2_dot = graph->CreateVarNode(var2_dot_desc);
  auto *new_transfer_layout_node = graph->CreateOpNode(&new_transfer_layout_desc);
  IR_NODE_LINK_TO(op_node, var2_dot);
  IR_NODE_LINK_TO(var2_dot, new_transfer_layout_node);
  IR_NODE_UNLINK(op_node, var2);
  IR_NODE_LINK_TO(new_transfer_layout_node, var2);

  for (auto var1 : op_node->inputs) {
    auto transfer_layout0_op = var1->inputs[0];
    auto var0 = transfer_layout0_op->inputs[0];
    IR_NODE_UNLINK(var0, transfer_layout0_op);
    //IR_NODE_UNLINK(var1, op_node);
    IR_NODE_LINK_TO(var0, op_node);

    op_node->Op()->RenameInput(var1->Name(), var0->Name());
    remove_nodes.emplace(transfer_layout0_op);
    remove_nodes.emplace(var1);
  }
  op_node->Op()->RenameOutput(var2->Name(), name);
  GraphSafeRemoveNodes(graph, remove_nodes);
}

bool TransferLayoutElimPass::InputAllTransferlayout(const ir::Node *op_node) const {
   
  for (auto var1 : op_node->inputs) {

    if (var1->inputs.size() != 1L) {
      return false;
    }
    if (var1->outputs.size() != 1L) {
      return false;
    }
    if (var1->inputs[0]->Name() != "transfer_layout") {
      return false;
    }
  }
  return true;
}

// (X) means deleted nodes
// (G) means generated node
//            var0              
//              |                         
//      transfer_layout0(X)          
//              |                       
//            var1      
//              |                                  
//      transfer_layout1(X ,op_node)                                   
//              |                                        
//             var2                                           
//         |    |     |                                             
//       op0   op1    op2
  
void TransferLayoutElimPass::ElimTwoTranferlayout(Node *op_node, ir::Graph* graph) const {
  std::unordered_set<const Node *> remove_nodes;
  auto var1 = op_node->inputs[0];
  auto transfer_layout0 = var1->inputs[0];
   auto var0 = transfer_layout0->inputs[0];
  auto var2 = op_node->outputs[0];
  remove_nodes.emplace(transfer_layout0);
  remove_nodes.emplace(var1);
  remove_nodes.emplace(op_node);
  remove_nodes.emplace(var2);

  for (auto next_op : var2->outputs)
  {
    IR_NODE_LINK_TO(var0, next_op);
    next_op->Op()->RenameInput(var2->Name(), var0->Name());  
  }
  GraphSafeRemoveNodes(graph, remove_nodes);
}

void TransferLayoutElimPass::ApplyImpl(ir::Graph* graph) const {

  const std::string pattern_name = "transfer_layout_elim";
  FusePassBase::Init(pattern_name, graph);
  
  while(1) {
    auto op_node_sorted = framework::ir::TopologyVarientSort(
        *graph, static_cast<framework::ir::SortKind>(0));
    bool modify = false;
    for (auto *op_node : op_node_sorted) {
      if (!op_node->IsOp()) continue;
      if (InputAllTransferlayout(op_node)) {
        std::cout << op_node->Name() << std::endl;
        if (op_node->Name() == "concat") {
          // 把concat的输入的tranfer_layout移动到concat之后喽！
          PutTranferlayoutAfterOp(op_node, graph);
          int axis = op_node->Op()->GetAttrIfExists<int>("axis");
          if (axis == 1) {
            op_node->Op()->SetAttr("axis", 3);
          }
          modify = true;
          break;
        }
        if (op_node->Name() == "pool2d" && 0) {
          PutTranferlayoutAfterOp(op_node, graph);
          op_node->Op()->SetAttr("data_format", std::string("NHWC"));
          modify = true;
          break;
        }
        if (op_node->Name() == "elementwise_add") {
          PutTranferlayoutAfterOp(op_node, graph);
          modify = true;
          break;
        }
        if (op_node->Name() == "hard_swish") {
          PutTranferlayoutAfterOp(op_node, graph);
          modify = true;
          break;
        }
        if (op_node->Name() == "transfer_layout") {
          ElimTwoTranferlayout(op_node, graph);
           modify = true;
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

REGISTER_PASS(transfer_layout_elim, paddle::framework::ir::TransferLayoutElimPass);
