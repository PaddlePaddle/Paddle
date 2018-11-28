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

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/inplace_op_pass.h"
#include "paddle/fluid/framework/details/op_graph_view.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

static VarHandle *FindUniqueInOutByVarName(ComputationOpHandle *op,
                                           const std::string &name,
                                           bool is_input,
                                           bool skip_persistable) {
  const std::string in_out_str = (is_input ? "inputs" : "outputs");
  std::string argument_name;
  if (is_input) {
    auto &inputs = op->Node()->Op()->Inputs();
    if (inputs.count(name) == 0 || inputs.at(name).empty()) return nullptr;
    PADDLE_ENFORCE(inputs.at(name).size() == 1, "Invalid inputs %s of op %s",
                   name, op->Name());
    argument_name = inputs.at(name)[0];
  } else {
    auto &outputs = op->Node()->Op()->Outputs();
    if (outputs.count(name) == 0 || outputs.at(name).empty()) return nullptr;
    PADDLE_ENFORCE(outputs.at(name).size() == 1, "Invalid outputs %s of op %s",
                   name, op->Name());
    argument_name = outputs.at(name)[0];
  }

  auto *in_outs = (is_input ? &(op->Inputs()) : &(op->Outputs()));

  VarHandle *ret = nullptr;
  for (auto *var_base : *in_outs) {
    auto *var = dynamic_cast<VarHandle *>(var_base);
    if (var == nullptr) continue;
    if (var->name_ == argument_name && var->scope_idx_ == op->GetScopeIdx()) {
      PADDLE_ENFORCE(ret == nullptr,
                     "Multiple %s with name %s(%s) in scope_idx %d of op %s",
                     in_out_str, name, argument_name, op->GetScopeIdx(),
                     op->DebugString());
      ret = var;
    }
  }
  PADDLE_ENFORCE_NOT_NULL(
      ret, "Cannot find %s with name %s(%s) in op %s(%d): %s", in_out_str, name,
      argument_name, op->Name(), op->GetScopeIdx(), op->DebugString());
  if (skip_persistable && ret->Node()->Var()->Persistable()) {
    return nullptr;
  } else {
    return ret;
  }
}

static VarHandle *FindUniqueInputByVarName(ComputationOpHandle *op,
                                           const std::string &name,
                                           bool skip_persistable = false) {
  return FindUniqueInOutByVarName(op, name, true, skip_persistable);
}

static VarHandle *FindUniqueOutputByVarName(ComputationOpHandle *op,
                                            const std::string &name,
                                            bool skip_persistable = false) {
  return FindUniqueInOutByVarName(op, name, false, skip_persistable);
}

struct VarHandleComparator {
  bool operator()(const VarHandle *var1, const VarHandle *var2) const {
    return var1->scope_idx_ != var2->scope_idx_
               ? var1->scope_idx_ < var2->scope_idx_
               : var1->name_ < var2->name_;
  }

  static bool IsEqual(const VarHandle *var1, const VarHandle *var2) {
    return var1->scope_idx_ == var2->scope_idx_ && var1->name_ == var2->name_;
  }
};

// Return read times or written times after var->GeneratedOp()
static std::pair<int, int> ReadWriteTimes(
    VarHandle *var, const OpGraphView &graph,
    const std::vector<OpHandleBase *> &all_ops) {
  std::pair<int, int> ret{0, 0};

  auto visitor = [&](OpHandleBase *pending_op) {
    for (auto *in : pending_op->Inputs()) {
      auto *in_var = dynamic_cast<VarHandle *>(in);
      if (in_var != nullptr && VarHandleComparator::IsEqual(in_var, var)) {
        VLOG(8) << "Read " << in_var->name_ << " in " << pending_op->Name();
        ++ret.first;
        break;
      }
    }

    for (auto *out : pending_op->Outputs()) {
      auto *out_var = dynamic_cast<VarHandle *>(out);
      if (out_var != nullptr && VarHandleComparator::IsEqual(out_var, var)) {
        VLOG(8) << "Write " << out_var->name_ << " in " << pending_op->Name();
        ++ret.second;
        break;
      }
    }
  };

  if (var->GeneratedOp()) {
    graph.VisitAllPendingOps(var->GeneratedOp(), visitor);
  } else {
    std::for_each(all_ops.begin(), all_ops.end(), visitor);
  }
  return ret;
}

std::unique_ptr<ir::Graph> InplaceOpPass::ApplyImpl(
    std::unique_ptr<ir::Graph> ir_graph) const {
  const auto all_ops = ir::FilterByNodeWrapper<OpHandleBase>(*ir_graph);
  const OpGraphView graph(all_ops);

  auto has_no_write = [&](VarHandle *var) -> bool {
    auto rw_times = ReadWriteTimes(var, graph, all_ops);
    VLOG(10) << var->name_ << " reads " << rw_times.first << " time(s), writes "
             << rw_times.second << " time(s)";
    return rw_times.second == 0;
  };

  std::set<VarHandle *, VarHandleComparator> non_modified_inplace_vars;

  for (auto &op : all_ops) {
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
    if (compute_op == nullptr) continue;
    OperatorBase &op_base = compute_op->GetOp();
    auto *inplace_ctx = op_base.InplaceContext();
    // std::string -> std::vector<std::string>
    auto inplace_var_map = op_base.GetNonModifiedInplaceVarMap();
    inplace_ctx->ClearNonModified();
    for (auto &var_pair : inplace_var_map) {
      VarHandle *in_var = FindUniqueInputByVarName(compute_op, var_pair.first);
      // Condition: in_var cannot be written afterwards
      if (in_var == nullptr || !has_no_write(in_var)) continue;

      std::set<VarHandle *, VarHandleComparator> out_vars_set;
      std::vector<std::string> can_inplace_vars;
      std::unordered_map<std::string, VarHandle *> out_var_name_handle_map;
      for (auto &out_var_name : var_pair.second) {
        VarHandle *out_var =
            FindUniqueOutputByVarName(compute_op, out_var_name);
        if (out_var != nullptr && out_vars_set.count(out_var) == 0 &&
            has_no_write(out_var)) {
          can_inplace_vars.push_back(out_var_name);
          out_var_name_handle_map[out_var_name] = out_var;
        }
      }

      if (!can_inplace_vars.empty()) {
        non_modified_inplace_vars.insert(in_var);
      }

      for (size_t i = 0; i < can_inplace_vars.size(); ++i) {
        const std::string &in_var_name = var_pair.first;
        const std::string &out_var_name = can_inplace_vars[i];
        inplace_ctx->UpdateNonModified(in_var_name, out_var_name);
        VLOG(10) << "Maybe non-modified inplace share buffer between Input("
                 << in_var_name << ") -> Output(" << out_var_name
                 << ") inside operator: " << op_base.DebugString();
        non_modified_inplace_vars.insert(
            out_var_name_handle_map.at(out_var_name));
      }
    }
  }

  auto do_not_appear = [&](VarHandle *var) -> bool {
    auto rw_times = ReadWriteTimes(var, graph, all_ops);
    VLOG(10) << var->name_ << " reads " << rw_times.first << " time(s), writes "
             << rw_times.second << " time(s)";
    return rw_times.first <= 1 && rw_times.second == 0;
  };

  for (auto &op : all_ops) {
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
    if (compute_op == nullptr) continue;
    OperatorBase &op_base = compute_op->GetOp();
    auto *inplace_ctx = op_base.InplaceContext();
    auto inplace_var_map = op_base.GetModifiedInplaceVarMap();
    inplace_ctx->ClearModified();
    for (auto &var_pair : inplace_var_map) {
      VarHandle *in_var =
          FindUniqueInputByVarName(compute_op, var_pair.first, true);
      if (in_var == nullptr || non_modified_inplace_vars.count(in_var) ||
          !do_not_appear(in_var)) {
        continue;
      }

      VarHandle *out_var =
          FindUniqueOutputByVarName(compute_op, var_pair.second);
      if (out_var == nullptr || non_modified_inplace_vars.count(out_var)) {
        continue;
      }

      bool has_multiple_out_var = std::any_of(
        compute_op->Outputs().begin(), compute_op->Outputs().end(),
        [out_var] (VarHandleBase *var_base) {
          auto var = dynamic_cast<VarHandle *>(var_base);
          return var != nullptr && var != out_var && VarHandleComparator::IsEqual(var, out_var);
        });

      if (!has_multiple_out_var) {
        inplace_ctx->UpdateModified(var_pair.first, var_pair.second);
      }
    }
  }

  return ir_graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_op_pass, paddle::framework::details::InplaceOpPass);
