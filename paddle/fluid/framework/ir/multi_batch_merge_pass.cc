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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
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

VarDesc* UpdateGradVarDesc(VarDesc* var_desc, int repeat,
                           const std::unordered_set<std::string> grad_names) {
  VLOG(3) << "update " << var_desc->Name() << " to repeat " << repeat;
  if (grad_names.find(var_desc->Name()) != grad_names.end()) {
    VarDesc* repeated_var = new VarDesc(*var_desc->Proto());
    std::string new_gname =
        string::Sprintf("%s.repeat.%d", repeated_var->Name(), repeat);
    repeated_var->SetName(new_gname);
    return repeated_var;
  }
  return var_desc;
}

std::unique_ptr<Graph> BatchMergePass::ApplyImpl(
    std::unique_ptr<Graph> graph) const {
  auto result = std::unique_ptr<Graph>(new Graph);
  result->ResetNodeId();
  int num_repeats = Get<const int>(kNumRepeats);
  std::vector<Node*> forward_backward_ops;
  std::vector<Node*> optimize_ops;  // record op_role != forward/backward
  std::unordered_set<std::string> grad_names;

  std::vector<ir::Node*> nodes = TopologySortOperations(*graph);

  // 1. record op nodes of different roles
  for (auto node : nodes) {
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

  // 2. copy forward backward
  ir::Node* prev_repeat_last_op_node = nullptr;
  // record origin_grad -> repeated grad list map.
  std::map<ir::Node*, std::vector<ir::Node*>> grad_repeated_map;
  for (int i = 0; i < num_repeats; ++i) {
    std::unordered_set<ir::Node*> copied;
    std::unordered_map<std::string, std::vector<ir::Node*>> created;
    for (size_t node_idx = 0; node_idx < forward_backward_ops.size();
         ++node_idx) {
      auto node = forward_backward_ops[node_idx];
      OpDesc* repeated_op = new OpDesc(*(node->Op()), node->Op()->Block());
      // 3. rename grad outputs to current repeat.
      for (auto outname : repeated_op->OutputArgumentNames()) {
        if (grad_names.find(outname) != grad_names.end()) {
          std::string new_gname = string::Sprintf("%s.repeat.%d", outname, i);
          repeated_op->RenameOutput(outname, new_gname);
        }
      }
      auto repeated_node = result->CreateOpNode(repeated_op);
      copied.insert(node);

      // 4. add deps between repeats
      if (node_idx == forward_backward_ops.size() - 1) {
        prev_repeat_last_op_node = repeated_node;
      }
      if (node_idx == 0 && prev_repeat_last_op_node) {
        auto* depvar = result->CreateControlDepVar();
        prev_repeat_last_op_node->outputs.push_back(depvar);
        depvar->inputs.push_back(prev_repeat_last_op_node);
        repeated_node->inputs.push_back(depvar);
        depvar->outputs.push_back(repeated_node);
      }

      for (auto in_node : node->inputs) {
        if (in_node->IsCtrlVar()) {
          continue;
        }
        ir::Node* var = nullptr;
        if (copied.find(in_node) == copied.end()) {
          var = result->CreateVarNode(
              UpdateGradVarDesc(in_node->Var(), i, grad_names));
          if (grad_names.find(in_node->Var()->Name()) != grad_names.end()) {
            grad_repeated_map[in_node].push_back(var);
          }
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
        if (copied.find(out_node) == copied.end()) {
          var = result->CreateVarNode(
              UpdateGradVarDesc(out_node->Var(), i, grad_names));
          if (grad_names.find(out_node->Var()->Name()) != grad_names.end()) {
            grad_repeated_map[out_node].push_back(var);
          }
          copied.insert(out_node);
          created[out_node->Name()].push_back(var);
        } else {
          var = created.at(out_node->Name()).back();
        }
        repeated_node->outputs.push_back(var);
        var->inputs.push_back(repeated_node);
      }
    }
  }

  // 5. create GRAD merge op node
  for (auto kv : grad_repeated_map) {
    OpDesc sum_op;
    sum_op.SetType("sum");
    std::vector<std::string> repeated_grad_names;
    for (auto r : kv.second) {
      repeated_grad_names.push_back(r->Var()->Name());
    }
    sum_op.SetInput("X", repeated_grad_names);
    sum_op.SetOutput("Out", {kv.first->Var()->Name()});
    sum_op.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                   static_cast<int>(OpRole::kBackward));
    auto sum_op_node = result->CreateOpNode(&sum_op);
    for (auto r : kv.second) {
      sum_op_node->inputs.push_back(r);
      r->outputs.push_back(sum_op_node);
    }
    auto sum_out_var_node = result->CreateVarNode(kv.first->Var());
    sum_op_node->outputs.push_back(sum_out_var_node);
    sum_out_var_node->inputs.push_back(sum_op_node);

    OpDesc scale_op;
    scale_op.SetType("scale");
    scale_op.SetInput("X", {sum_out_var_node->Var()->Name()});
    // NOTE: inplace scale.
    scale_op.SetOutput("Out", {sum_out_var_node->Var()->Name()});
    scale_op.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                     static_cast<int>(OpRole::kBackward));
    auto scale_op_node = result->CreateOpNode(&scale_op);
    scale_op_node->inputs.push_back(sum_out_var_node);
    sum_out_var_node->outputs.push_back(scale_op_node);
    auto scale_out_var_node = result->CreateVarNode(sum_out_var_node->Var());
    scale_op_node->outputs.push_back(scale_out_var_node);
    scale_out_var_node->inputs.push_back(scale_op_node);
  }
  // 6. add optimize ops
  // 7. release original op_descs (forward/backward)
  result->ResolveHazard();
  return result;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_batch_merge_pass, paddle::framework::ir::BatchMergePass)
    .RequirePassAttr(paddle::framework::ir::kNumRepeats);
