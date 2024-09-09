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

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle::framework::ir {

static const char kNumRepeats[] = "num_repeats";  // NOLINT
typedef std::unordered_map<std::string, std::vector<ir::Node*>> SSAVarList;

ir::Node* SameNameVar(std::unordered_set<ir::Node*> all, ir::Node* target) {
  for (auto n : all) {
    if (target->IsVar() && target->Name() == n->Name()) {
      return n;
    }
  }
  return nullptr;
}

VarDesc CopyVarDesc(VarDesc* var_desc) {
  VarDesc repeated_var(var_desc->Name());
  // copy other variable attributes
  if (var_desc->GetType() != proto::VarType::READER) {
    repeated_var.SetType(var_desc->GetType());
    repeated_var.SetShape(var_desc->GetShape());
    repeated_var.SetDataType(var_desc->GetDataType());
    repeated_var.SetLoDLevel(var_desc->GetLoDLevel());
    repeated_var.SetPersistable(var_desc->Persistable());
  } else {
    // TODO(typhoonzero): copy reader var
  }
  return repeated_var;
}

VarDesc UpdateGradVarDesc(
    VarDesc* var_desc,
    int repeat,
    const std::unordered_set<std::string>& grad_names,
    const std::unordered_set<std::string>& bn_vars_need_rename) {
  if (grad_names.find(var_desc->Name()) != grad_names.end() ||
      bn_vars_need_rename.find(var_desc->Name()) != bn_vars_need_rename.end()) {
    std::string new_gname =
        string::Sprintf("%s.repeat.%d", var_desc->Name(), repeat);
    VarDesc repeated_var = CopyVarDesc(var_desc);
    repeated_var.SetName(new_gname);
    VLOG(3) << "update " << var_desc->Name() << " to repeat " << repeat;
    return repeated_var;
  }
  return *var_desc;
}

void BatchMergePass::ApplyImpl(ir::Graph* graph) const {
  int num_repeats = Get<const int>(kNumRepeats);
  std::vector<Node*> forward_backward_ops;
  std::vector<Node*> optimize_ops;
  std::vector<Node*> lr_ops;  // ops other than forward/backward/optimize
  std::unordered_set<std::string> grad_names;
  std::unordered_map<std::string, std::string> gradname2paramname;

  std::vector<ir::Node*> nodes = TopologySortOperations(*graph);
  auto origin_nodes = graph->ReleaseNodes();
  VLOG(3) << "origin nodes count: " << origin_nodes.size();
  ir::Graph& result = *graph;

  // 1. record op nodes of different roles
  for (auto node : nodes) {
    if (!node->IsOp()) continue;
    PADDLE_ENFORCE_NOT_NULL(
        node->Op(),
        common::errors::InvalidArgument("Node(%s) must hold op description.",
                                        node->Name()));
    int op_role = PADDLE_GET_CONST(
        int,
        node->Op()->GetAttr(
            framework::OpProtoAndCheckerMaker::OpRoleAttrName()));
    if ((op_role == static_cast<int>(framework::OpRole::kForward)) ||
        (op_role & static_cast<int>(framework::OpRole::kBackward)) ||
        (op_role & static_cast<int>(framework::OpRole::kLoss))) {
      forward_backward_ops.push_back(node);
    } else if ((op_role & static_cast<int>(framework::OpRole::kOptimize)) ||
               (op_role & static_cast<int>(framework::OpRole::kDist)) ||
               (op_role & static_cast<int>(framework::OpRole::kRPC))) {
      optimize_ops.push_back(node);
      auto op_role_var = node->Op()->GetNullableAttr(
          OpProtoAndCheckerMaker::OpRoleVarAttrName());
      auto op_role_vars =
          PADDLE_GET_CONST(std::vector<std::string>, op_role_var);
      for (size_t i = 0; i < op_role_vars.size(); i += 2) {
        grad_names.insert(op_role_vars[i + 1]);
        gradname2paramname[op_role_vars[i + 1]] = op_role_vars[i];
      }
    } else if (op_role & static_cast<int>(framework::OpRole::kLRSched)) {
      lr_ops.push_back(node);
    } else {  // NOLINT
      PADDLE_THROW(
          common::errors::InvalidArgument("Invalid op role(%d), in node(%s).",
                                          static_cast<int>(op_role),
                                          node->Name()));
    }
  }

  // 2. copy forward backward
  ir::Node* prev_repeat_last_op_node = nullptr;
  // record origin_grad -> repeated_grad_list map.
  std::map<ir::Node*, std::vector<ir::Node*>> grad_repeated_map;
  std::map<std::string, std::vector<ir::Node*>> created;
  std::unordered_set<std::string> bn_vars_need_rename;
  for (int i = 0; i < num_repeats; ++i) {
    std::unordered_set<ir::Node*> copied;
    for (size_t node_idx = 0; node_idx < forward_backward_ops.size();
         ++node_idx) {
      auto node = forward_backward_ops[node_idx];
      OpDesc repeated_op(*(node->Op()), node->Op()->Block());
      // 3. rename grad outputs to current repeat.
      for (auto const& outname : repeated_op.OutputArgumentNames()) {
        if (grad_names.find(outname) != grad_names.end()) {
          std::string new_gname = string::Sprintf("%s.repeat.%d", outname, i);
          repeated_op.RenameOutput(outname, new_gname);
          // remove op_role_var for backward ops that outputs grad for a
          // parameter.
          repeated_op.SetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName(),
                              std::vector<std::string>());
        }
      }
      // 3.5 let batch_norm ops use independent vars, note batch_norm_grad do
      // not need this update, because only moving mean and variance should be
      // differ, trainable parameter scale and bias is the same as other
      // parameters.
      if (node->Name() == "batch_norm") {
        // NOTE: assume bn op created by layers use save var as output mean and
        // variance
        std::string new_mean_name =
            string::Sprintf("%s.repeat.%d", repeated_op.Input("Mean")[0], i);
        std::string new_var_name = string::Sprintf(
            "%s.repeat.%d", repeated_op.Input("Variance")[0], i);
        bn_vars_need_rename.insert(repeated_op.Input("Mean")[0]);
        bn_vars_need_rename.insert(repeated_op.Input("Variance")[0]);
        VLOG(3) << "renaming " << repeated_op.Input("Mean")[0] << " to "
                << new_mean_name;
        repeated_op.RenameInput(repeated_op.Input("Mean")[0], new_mean_name);
        repeated_op.RenameInput(repeated_op.Input("Variance")[0], new_var_name);
        repeated_op.RenameOutput(repeated_op.Output("MeanOut")[0],
                                 new_mean_name);
        repeated_op.RenameOutput(repeated_op.Output("VarianceOut")[0],
                                 new_var_name);
      }

      // 3.9 do copy
      auto repeated_node = result.CreateOpNode(&repeated_op);
      copied.insert(node);

      // 4. add deps between repeats
      if (node_idx == forward_backward_ops.size() - 1) {
        prev_repeat_last_op_node = repeated_node;
      }
      if (node_idx == 0 && prev_repeat_last_op_node) {
        auto* depvar = result.CreateControlDepVar();
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
        auto updated_var = UpdateGradVarDesc(
            in_node->Var(), i, grad_names, bn_vars_need_rename);
        // should be initialized by startup, how to initialize tensor in the
        // scope?
        if (node->Name() == "batch_norm" &&
            bn_vars_need_rename.find(in_node->Name()) !=
                bn_vars_need_rename.end()) {
          // Create bn mean/variance for each repeat
          var = result.CreateVarNode(&updated_var);
          created[updated_var.Name()].push_back(var);
          copied.insert(in_node);
          repeated_node->inputs.push_back(var);
          var->outputs.push_back(repeated_node);
          continue;
        }

        // for other ops
        if (in_node->inputs.empty() && i > 0) {
          // do not copy head vars (inputs, params) in repeats > 0
          var = created.at(in_node->Name()).back();
        } else {
          if (copied.find(in_node) == copied.end()) {
            var = result.CreateVarNode(&updated_var);
            if (grad_names.find(in_node->Var()->Name()) != grad_names.end()) {
              grad_repeated_map[in_node].push_back(var);
            }
            copied.insert(in_node);
            created[updated_var.Name()].push_back(var);
          } else {
            var = created.at(updated_var.Name()).back();
          }
        }
        repeated_node->inputs.push_back(var);
        var->outputs.push_back(repeated_node);
      }
      for (auto out_node : node->outputs) {
        if (out_node->IsCtrlVar()) {
          continue;
        }
        ir::Node* var = nullptr;
        auto updated_var = UpdateGradVarDesc(
            out_node->Var(), i, grad_names, bn_vars_need_rename);
        if (copied.find(out_node) == copied.end()) {
          var = result.CreateVarNode(&updated_var);
          if (grad_names.find(out_node->Var()->Name()) != grad_names.end()) {
            grad_repeated_map[out_node].push_back(var);
          }
          copied.insert(out_node);
          created[updated_var.Name()].push_back(var);
        } else {
          var = created.at(updated_var.Name()).back();
        }
        repeated_node->outputs.push_back(var);
        var->inputs.push_back(repeated_node);
      }
    }
  }  // end copy forward backward

  // 5. create GRAD merge op node: sum(repeat.0...repeat.n) ->
  // scale(1/num_repeats)
  for (auto const& kv : grad_repeated_map) {
    OpDesc sum_op;
    sum_op.SetType("sum");
    std::vector<std::string> repeated_grad_names;
    std::vector<std::string> param_grad_op_role_var;
    repeated_grad_names.reserve(kv.second.size());
    for (auto r : kv.second) {
      repeated_grad_names.push_back(r->Var()->Name());
    }
    // NOTE: use op_role_var to control allreduce op appending in
    //       multi_devices_graph_pass, we want to append op_role_var
    //       only once for the merged gradient, so break after first call.
    param_grad_op_role_var.push_back(
        gradname2paramname.at(kv.first->Var()->Name()));        // param
    param_grad_op_role_var.push_back(kv.first->Var()->Name());  // grad

    sum_op.SetInput("X", repeated_grad_names);
    sum_op.SetOutput("Out", {kv.first->Var()->Name()});
    sum_op.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                   static_cast<int>(OpRole::kBackward));
    auto sum_op_node = result.CreateOpNode(&sum_op);
    for (auto r : kv.second) {
      sum_op_node->inputs.push_back(r);
      r->outputs.push_back(sum_op_node);
    }
    auto sum_out_var_node = result.CreateVarNode(kv.first->Var());
    sum_op_node->outputs.push_back(sum_out_var_node);
    sum_out_var_node->inputs.push_back(sum_op_node);
    created[sum_out_var_node->Name()].push_back(sum_out_var_node);

    OpDesc scale_op;
    scale_op.SetType("scale");
    scale_op.SetInput("X", {sum_out_var_node->Var()->Name()});
    // NOTE: inplace scale.
    scale_op.SetOutput("Out", {sum_out_var_node->Var()->Name()});
    scale_op.SetAttr(
        "scale", static_cast<float>(1.0f / static_cast<float>(num_repeats)));
    scale_op.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
                     static_cast<int>(OpRole::kBackward));

    scale_op.SetAttr(OpProtoAndCheckerMaker::OpRoleVarAttrName(),
                     param_grad_op_role_var);

    auto scale_op_node = result.CreateOpNode(&scale_op);
    scale_op_node->inputs.push_back(sum_out_var_node);
    sum_out_var_node->outputs.push_back(scale_op_node);
    auto scale_out_var_node = result.CreateVarNode(sum_out_var_node->Var());
    scale_op_node->outputs.push_back(scale_out_var_node);
    scale_out_var_node->inputs.push_back(scale_op_node);
    created[scale_out_var_node->Name()].push_back(scale_out_var_node);
  }
  // 6. add optimize ops
  {
    auto copy_node = [&result, &created](ir::Node* node) {
      auto op_node = result.CreateOpNode(node->Op());
      // copy op ins/outs
      // NOTE: for send/recv ops, the OpDesc uses ctrldepvar to describe
      // dependencies, so create those depvars if OpDesc have in/outs.
      for (auto in_node : node->inputs) {
        if (in_node->IsCtrlVar() && !in_node->Var()) {
          continue;
        }
        ir::Node* var = nullptr;
        if (created.find(in_node->Name()) == created.end()) {
          var = result.CreateVarNode(in_node->Var());
          created[in_node->Name()].push_back(var);
        } else {
          var = created.at(in_node->Name()).back();
        }
        op_node->inputs.push_back(var);
        var->outputs.push_back(op_node);
      }
      for (auto out_node : node->outputs) {
        if (out_node->IsCtrlVar() && !out_node->Var()) {
          continue;
        }
        auto var = result.CreateVarNode(out_node->Var());
        created[out_node->Name()].push_back(var);
        op_node->outputs.push_back(var);
        var->inputs.push_back(op_node);
      }
    };
    for (auto node : lr_ops) {
      copy_node(node);
    }
    for (auto node : optimize_ops) {
      copy_node(node);
    }
  }
}

}  // namespace paddle::framework::ir

REGISTER_PASS(multi_batch_merge_pass, paddle::framework::ir::BatchMergePass)
    .RequirePassAttr(paddle::framework::ir::kNumRepeats);
