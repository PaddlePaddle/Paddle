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

#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

// Each ShareTensorBufferOpHandle should only have one pending
// ComputationOpHandle
static details::ComputationOpHandle *GetUniquePendingComputationOpHandle(
    details::ShareTensorBufferOpHandle *share_tensor_op) {
  details::ComputationOpHandle *result_op = nullptr;
  for (Node *out_var : share_tensor_op->Node()->outputs) {
    for (Node *pending_op : out_var->outputs) {
      auto &op = pending_op->Wrapper<details::OpHandleBase>();
      auto *compute_op = dynamic_cast<details::ComputationOpHandle *>(&op);
      PADDLE_ENFORCE_NOT_NULL(compute_op);

      if (result_op == nullptr) {
        result_op = compute_op;
      } else {
        PADDLE_ENFORCE_EQ(result_op, compute_op);
      }
    }
  }

  PADDLE_ENFORCE_NOT_NULL(result_op);
  return result_op;
}

void MemoryReusePass::ApplyImpl(Graph *graph) const {
  graph_ = graph;
  all_vars_ = &(graph_->Get<details::GraphVars>(details::kGraphVars));
  var_infos_ = &(Get<MemOptVarInfoMapList>(kMemOptVarInfoMapList));
  last_live_ops_of_vars_ =
      &(Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars));

  reused_var_names_.resize(all_vars_->size());
  var_descs_.resize(all_vars_->size());

  // Collect the existing ShareTensorBufferOpHandles.
  // This is because (1) we want to reuse the existing
  // ShareTensorBufferOpHandles to avoid inserting too many ops;
  // (2) what is more important, a variable cannot be reused
  // by two different variables, which may cause wrong calculation
  // results. We have to know which variables have been reused.
  CollectShareTensorBufferOpHandles();
  CollectReusedVars();
  Run(graph);

  std::map<size_t, size_t> op_num;
  for (auto &pair : ops_) {
    ++op_num[pair.first->GetScopeIdx()];
  }

  for (auto &pair : op_num) {
    VLOG(2) << "Create " << pair.second
            << " ShareTensorBufferOpHandles in Scope " << pair.first;
  }
}

bool MemoryReusePass::TryReuseVar(details::VarHandle *in_var,
                                  details::VarHandle *out_var) const {
  auto *op =
      dynamic_cast<details::ComputationOpHandle *>(out_var->GeneratedOp());
  PADDLE_ENFORCE_NOT_NULL(op);
  if (IsVarsReusable(in_var, out_var)) {
    AddReuseVar(op, in_var, out_var);
    return true;
  } else {
    return false;
  }
}

std::unordered_set<Node *> MemoryReusePass::FindNodesByName(
    const std::string &name, const std::vector<Node *> &nodes) const {
  std::unordered_set<ir::Node *> ret;
  for (auto *node : nodes) {
    if (node->Name() == name) {
      ret.insert(node);
    }
  }
  return ret;
}

VarDesc *MemoryReusePass::GetVarDesc(details::VarHandle *var) const {
  auto iter = var_descs_[var->scope_idx()].find(var->Name());
  if (iter == var_descs_[var->scope_idx()].end()) {
    PADDLE_ENFORCE((*all_vars_)[var->scope_idx()].count(var->Name()),
                   "Variable %s not found", var->Name());
    auto *desc =
        TryGetLatestVarDesc((*all_vars_)[var->scope_idx()].at(var->Name()));
    PADDLE_ENFORCE_NOT_NULL(desc);
    var_descs_[var->scope_idx()].emplace(var->Name(), desc);
    return desc;
  } else {
    return iter->second;
  }
}

void MemoryReusePass::CollectShareTensorBufferOpHandles() const {
  auto all_ops = FilterByNodeWrapper<details::OpHandleBase>(*graph_);
  for (auto *op : all_ops) {
    auto *share_buffer_op =
        dynamic_cast<details::ShareTensorBufferOpHandle *>(op);
    if (share_buffer_op != nullptr) {
      auto *compute_op = GetUniquePendingComputationOpHandle(share_buffer_op);
      PADDLE_ENFORCE(ops_.count(compute_op) == 0);
      ops_.emplace(compute_op, share_buffer_op);
    }
  }
}

void MemoryReusePass::CollectReusedVars() const {
  for (auto &pair : ops_) {
    auto reused_vars = pair.second->ReusedVarSet();
    reused_var_names_[pair.first->GetScopeIdx()].insert(reused_vars.begin(),
                                                        reused_vars.end());
  }
}

bool MemoryReusePass::IsVarAlreadyReused(details::VarHandle *var) const {
  return reused_var_names_[var->scope_idx()].count(var->Name()) > 0;
}

details::ShareTensorBufferOpHandle *
MemoryReusePass::InsertShareTensorBufferOpHandleToGraph(
    details::ComputationOpHandle *op) const {
  auto *buffer_share_node =
      graph_->CreateEmptyNode("buffer_share", ir::Node::Type::kOperation);

  auto *buffer_share_op = new details::ShareTensorBufferOpHandle(
      buffer_share_node, op->GetScope(), op->GetScopeIdx(), op->GetOp()->Type(),
      {}, {});

  buffer_share_op->SetDeviceContext(
      op->GetPlace(),
      platform::DeviceContextPool::Instance().Get(op->GetPlace()));

  // Inputs of `buffer_share_op` should be all inputs of `op`
  for (auto *in_var : op->Inputs()) {
    buffer_share_op->AddInput(in_var);
  }

  // Add a dep_var to resolve write-after-write data hazard between
  // `buffer_share_op` and `op`.
  auto *dep_var = new details::DummyVarHandle(graph_->CreateControlDepVar());
  graph_->Get<details::GraphDepVars>(details::kGraphDepVars).emplace(dep_var);
  op->AddInput(dep_var);
  buffer_share_op->AddOutput(dep_var);

  ops_.emplace(op, buffer_share_op);
  return buffer_share_op;
}

bool MemoryReusePass::IsVarsReusable(details::VarHandle *in_var,
                                     details::VarHandle *out_var) const {
  const auto in_name = in_var->Name();
  const auto out_name = out_var->Name();

  if (in_name == out_name) {
    return false;
  }

  if (in_name == kEmptyVarName || out_name == kEmptyVarName) {
    return false;
  }

  if (IsVarAlreadyReused(in_var)) {
    return false;
  }

  // out_var must be the first version!!!
  auto out_var_iter = (*all_vars_)[out_var->scope_idx()].find(out_name);
  PADDLE_ENFORCE(out_var_iter != (*all_vars_)[out_var->scope_idx()].end() &&
                     !out_var_iter->second.empty(),
                 "Cannot find variable %s", out_name);

  if (out_var_iter->second[0] != out_var) {
    return false;
  }

  const VarDesc *in_var_desc = GetVarDesc(in_var);
  const VarDesc *out_var_desc = GetVarDesc(out_var);

  if (in_var_desc->Persistable() || out_var_desc->Persistable()) {
    return false;
  }

  if (in_var_desc->GetType() != proto::VarType::LOD_TENSOR ||
      out_var_desc->GetType() != proto::VarType::LOD_TENSOR) {
    return false;
  }

  if (!FindNodesByName(in_name, out_var->GeneratedOp()->Node()->outputs)
           .empty()) {
    return false;
  }

  if (!FindNodesByName(out_name, out_var->GeneratedOp()->Node()->inputs)
           .empty()) {
    return false;
  }

  auto all_input_args =
      out_var->GeneratedOp()->Node()->Op()->InputArgumentNames();
  if (std::count(all_input_args.begin(), all_input_args.end(), in_name) > 1) {
    return false;
  }

  return true;
}

void MemoryReusePass::AddReuseVar(details::ComputationOpHandle *op,
                                  details::VarHandle *in_var,
                                  details::VarHandle *out_var) const {
  PADDLE_ENFORCE((*var_infos_)[op->GetScopeIdx()].count(in_var->Name()) > 0,
                 "%s does not in mem-opt var infos", in_var->Name());

  if (ops_.count(op) == 0) {
    InsertShareTensorBufferOpHandleToGraph(op);
  }

  auto *share_buffer_op = ops_[op];

  auto &all_input_vars = share_buffer_op->Inputs();
  bool has_input = std::find(all_input_vars.begin(), all_input_vars.end(),
                             in_var) != all_input_vars.end();

  if (!has_input) {
    share_buffer_op->AddInput(in_var);
  }

  share_buffer_op->Add(
      (*var_infos_)[op->GetScopeIdx()].at(in_var->Name()).get(),
      out_var->Name());
  reused_var_names_[op->GetScopeIdx()].insert(in_var->Name());

  UpdateLastLiveOpOfVar(op, in_var, out_var);
}

// 1. Set last living op of in_var to be any last living op of out_var
// 2. Set reference count of in_var to be 1
void MemoryReusePass::UpdateLastLiveOpOfVar(details::ComputationOpHandle *op,
                                            details::VarHandle *in_var,
                                            details::VarHandle *out_var) const {
  size_t scope_idx = op->GetScopeIdx();
  auto out_var_op_iter =
      (*last_live_ops_of_vars_)[scope_idx].find(out_var->Name());
  PADDLE_ENFORCE(out_var_op_iter != (*last_live_ops_of_vars_)[scope_idx].end(),
                 "Cannot find variable %s", out_var->Name());
  PADDLE_ENFORCE(!out_var_op_iter->second.empty());

  auto &last_live_ops_of_in_var =
      (*last_live_ops_of_vars_)[scope_idx][in_var->Name()];
  last_live_ops_of_in_var.clear();
  last_live_ops_of_in_var.insert(*(out_var_op_iter->second.begin()));

  auto in_var_info_iter = (*var_infos_)[scope_idx].find(in_var->Name());
  PADDLE_ENFORCE(in_var_info_iter != (*var_infos_)[scope_idx].end(),
                 "Cannot find variable %s", in_var->Name());

  in_var_info_iter->second->SetRefCnt(1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
