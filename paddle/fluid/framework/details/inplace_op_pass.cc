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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/inplace_op_pass.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_handle_graph.h"

namespace paddle {
namespace framework {
namespace details {

static VarHandle *FindUniqueInOutByVarName(ComputationOpHandle *op,
                                           const std::string &name,
                                           bool is_input) {
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
  return ret->Node()->Var()->Persistable() ? nullptr : ret;
}

static VarHandle *FindUniqueInputByVarName(ComputationOpHandle *op,
                                           const std::string &name) {
  return FindUniqueInOutByVarName(op, name, true);
}

static VarHandle *FindUniqueOutputByVarName(ComputationOpHandle *op,
                                            const std::string &name) {
  return FindUniqueInOutByVarName(op, name, false);
}

struct ReadWriteCounter {
  inline void Add(const ReadWriteCounter &other) {
    read_cnt_ += other.read_cnt_;
    write_cnt_ += other.write_cnt_;
  }

  inline bool NoReadWrite() const { return read_cnt_ == 0 && write_cnt_ == 0; }

  inline bool OnlyWrite() const { return read_cnt_ == 0 && write_cnt_ > 0; }

  inline bool NoRead() const { return read_cnt_ == 0; }

  inline bool NoWrite() const { return write_cnt_ == 0; }

  size_t read_cnt_{0};
  size_t write_cnt_{0};
};

static void UpdateReadWriteCounter(
    OpHandleBase *op,
    std::unordered_map<
        size_t, std::unordered_map<std::string, ReadWriteCounter>> *counters) {
  auto update_handle = [op, counters](VarHandleBase *var_base, bool is_input) {
    auto *var = dynamic_cast<VarHandle *>(var_base);
    if (var == nullptr) return;
    if (counters->count(var->scope_idx_) &&
        counters->at(var->scope_idx_).count(var->name_)) {
      auto &cnt = counters->at(var->scope_idx_).at(var->name_);
      is_input ? ++cnt.read_cnt_ : ++cnt.write_cnt_;
    }
  };

  for (auto *in : op->Inputs()) {
    update_handle(in, true);
  }

  for (auto *out : op->Outputs()) {
    update_handle(out, false);
  }
}

std::unique_ptr<ir::Graph> InplaceOpPass::ApplyImpl(
    std::unique_ptr<ir::Graph> ir_graph) const {
  auto &all_ops = ir_graph->Get<GraphOps>(kGraphOps);
  OpHandleGraph graph(all_ops);
  for (auto &op : all_ops) {
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op.get());
    if (compute_op == nullptr) continue;
    auto &op_base = compute_op->GetOp();
    op_base.ClearInplaceWithNoChangeMap();
    auto possible_in_out = op_base.GetInplaceWithNoChangeMap();
    for (auto &pair : possible_in_out) {
      const auto &in_name = pair.first;
      auto *in_var = FindUniqueInputByVarName(compute_op, in_name);
      if (in_var == nullptr) continue;
      std::vector<VarHandle *> out_vars;
      std::vector<std::string> out_names;
      out_vars.reserve(pair.second.size());
      out_names.reserve(pair.second.size());
      for (auto &out_name : pair.second) {
        auto *out_var = FindUniqueOutputByVarName(compute_op, out_name);
        if (out_var == nullptr) continue;
        if (in_var->scope_idx_ != out_var->scope_idx_ ||
            in_var->name_ != out_var->name_) {
          out_vars.push_back(out_var);
          out_names.push_back(out_name);
        }
      }

      if (out_vars.empty()) continue;

      std::unordered_map<size_t,
                         std::unordered_map<std::string, ReadWriteCounter>>
          counters;
      // init counters
      auto &in_counter = counters[in_var->scope_idx_][in_var->name_];
      for (auto &out_var : out_vars) {
        counters[out_var->scope_idx_][out_var->name_];
      }

      // Find the op which generates Input(in_name)
      auto *generated_op = in_var->GeneratedOp();
      if (generated_op != nullptr) {
        auto pending_ops = graph.AllPendingOps(generated_op);
        for (auto &each_level_ops : pending_ops) {
          for (auto *pending_op : each_level_ops) {
            if (pending_op == op.get()) continue;  // skip current op
            UpdateReadWriteCounter(pending_op, &counters);
          }
        }
      } else {
        // FIXME(zjl): maybe pretty slow when generated_op is nullptr
        for (auto &tmp_op : all_ops) {
          if (tmp_op.get() == op.get()) continue;
          if (auto *tmp_compute_op =
                  dynamic_cast<ComputationOpHandle *>(op.get())) {
            UpdateReadWriteCounter(tmp_compute_op, &counters);
          }
        }
      }

      for (size_t i = 0; i < out_vars.size(); ++i) {
        auto *out_var = out_vars[i];
        auto &out_counter = counters[out_var->scope_idx_][out_var->name_];
        /*
        if ((in_counter.NoWrite() && out_counter.NoWrite()) ||
            (in_counter.NoReadWrite() && out_counter.OnlyWrite()) ||
            (in_counter.OnlyWrite() && out_counter.NoReadWrite())) {
        */
        if ((in_counter.NoWrite() && out_counter.NoWrite()) ||
            in_counter.NoReadWrite()) {
          // can inplace share without change
          VLOG(10) << "Inplace enabled: " << out_names[i] << " shares "
                   << in_name << " in op: " << op_base.DebugString()
                   << " of scope_idx = " << compute_op->GetScopeIdx();
          op_base.UpdateInplaceWithNoChangeMap(in_name, out_names[i]);
          in_counter.Add(out_counter);
        }
      }
    }
  }
  return ir_graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_op_pass, paddle::framework::details::InplaceOpPass);
