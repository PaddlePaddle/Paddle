//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/multi_batch_merge_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

static const char kNumRepeats[] = "num_repeats";

ir::Node* SameNameVar(std::unordered_set<ir::Node*> all, ir::Node* target) {
  for (auto n : all) {
    if (target->IsVar() && target->Name() == n->Name()) {
      return n;
    }
  }
  return nullptr;
}

std::unique_ptr<Graph> BatchMergePass::ApplyImpl(
    std::unique_ptr<Graph> graph) const {
  auto result = std::unique_ptr<Graph>(new Graph);
  int num_repeats = Get<const int>(kNumRepeats);
  std::vector<Node*> forward_backward_ops;
  std::vector<Node*> optimize_ops;  // record op_role != forward/backward
  std::unordered_set<std::string> grad_names;

  for (auto node : graph->Nodes()) {
    if (node->IsVar()) continue;
    int op_role = boost::get<int>(node->Op()->GetAttr(
        framework::OpProtoAndCheckerMaker::OpRoleAttrName()));
    if (op_role == static_cast<int>(framework::OpRole::kForward) ||
        op_role == static_cast<int>(framework::OpRole::kBackward)) {
      forward_backward_ops.push_back(node);
    } else if (op_role == static_cast<int>(framework::OpRole::kOptimize)) {
      optimize_ops.push_back(node);
      auto op_role_var =
          node->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName());
      grad_names.insert(boost::get<std::vector<std::string>>(op_role_var)[1]);
    }
  }

  // copy forward backward
  for (int i = 0; i < num_repeats; ++i) {
    std::unordered_set<ir::Node*> copied;
    // same var name may have many nodes in ssa with versions.
    std::unordered_map<std::string, std::vector<ir::Node*>> created;
    for (auto node : forward_backward_ops) {
      OpDesc* repeated_op = new OpDesc(*(node->Op()), node->Op()->Block());
      // rename grad outputs to current repeat.
      for (auto outname : repeated_op->OutputArgumentNames()) {
        if (grad_names.find(outname) != grad_names.end()) {
          std::string new_gname = string::Sprintf(outname, ".repeat", i);
          repeated_op->RenameOutput(outname, new_gname);
        }
      }
      auto repeated_node = result->CreateOpNode(repeated_op);
      copied.insert(node);
      for (auto in_node : node->inputs) {
        if (in_node->IsCtrlVar()) {
          continue;
        }
        ir::Node* var = nullptr;
        if (copied.find(in_node) == copied.end()) {
          var = result->CreateVarNode(in_node->Var());
          copied.insert(in_node);
          created[in_node->Name()].push_back(var);
        } else {
          var = created.at(in_node->Name()).back();
        }
        repeated_node->inputs.push_back(var);
        var->outputs.push_back(repeated_node);
      }
      for (auto out_node : node->outputs) {
        if (out_node->IsCtrlVar()) {
          continue;
        }
        ir::Node* var = nullptr;
        // if (copied.find(out_node) == copied.end()) {
        var = result->CreateVarNode(out_node->Var());
        copied.insert(out_node);
        created[out_node->Name()].push_back(var);
        // } else {
        //   var = created.at(out_node->Name()).back();
        // }
        repeated_node->outputs.push_back(var);
        var->inputs.push_back(repeated_node);
      }
    }
  }

  // 5. create GRAD merge op node, input per repeat GRAD var, output origin GRAD
  // var
  // 6. add optimize ops
  // 7. release original op_descs (forward/backward)
  return result;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_batch_merge_pass, paddle::framework::ir::BatchMergePass)
    .RequirePassAttr(paddle::framework::ir::kNumRepeats);
