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

#include "paddle/fluid/framework/details/fuse_all_reduce_graph_builder.h"
#include <list>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

static int GetUniqNum() {
  static int id = -1;
  return ++id;
}

std::unique_ptr<SSAGraph> FuseAllReduceGraphBuilder::Build(
    const ProgramDesc &program) const {
  auto graph = builder_->Build(program);

  auto all_reduce_ops =
      GetNotDependedAllReduceOp(graph.get(), program.Block(0));
  VLOG(10) << "All reduce op group number: " << all_reduce_ops.size();
  for (auto &op_group : all_reduce_ops) {
    FuseAllReduceOp(graph.get(), std::move(op_group), program.Block(0));
  }
  return graph;
}

inline static void GetAllInputOp(OpHandleBase *op,
                                 std::unordered_set<OpHandleBase *> *set) {
  for (auto &var : op->Inputs()) {
    auto *op = var->generated_op_;
    if (op == nullptr) {
      continue;
    }

    if (set->count(op)) {
      continue;
    }

    set->emplace(op);
    GetAllInputOp(op, set);
  }
}

inline static bool IsAnyOfParentInCurrentOp(
    OpHandleBase *op, const std::unordered_set<size_t> &cur_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map) {
  std::unordered_set<OpHandleBase *> inputs;
  inputs.emplace(op);
  GetAllInputOp(op, &inputs);

  for (auto *tmp : inputs) {
    auto it = offset_map.find(tmp);
    if (it == offset_map.end()) {
      continue;
    }
    if (cur_ops.count(it->second)) {
      return true;
    }
  }
  return false;
}

inline static void GetAllOutputOp(OpHandleBase *op,
                                  std::unordered_set<OpHandleBase *> *set) {
  for (auto &var : op->Outputs()) {
    for (auto *op : var->pending_ops_) {
      if (op == nullptr) {
        continue;
      }
      if (set->count(op)) {
        continue;
      }
      set->emplace(op);
      GetAllOutputOp(op, set);
    }
  }
}

inline static bool IsAnyOfChildInCurrentOp(
    OpHandleBase *op, const std::unordered_set<size_t> &cur_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map) {
  std::unordered_set<OpHandleBase *> set;
  set.emplace(op);
  GetAllOutputOp(op, &set);
  for (auto *tmp : set) {
    auto it = offset_map.find(tmp);
    if (it == offset_map.end()) {
      continue;
    }
    if (cur_ops.count(it->second)) {
      return true;
    }
  }
  return false;
}

inline static void ResolveOpDeps(
    const std::vector<std::unique_ptr<OpHandleBase>> &all_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map,
    std::list<std::unordered_set<size_t>>::iterator cur_it,
    std::list<std::unordered_set<size_t>> *res) {
  std::unordered_set<size_t> before_deps;
  std::unordered_set<size_t> cur_ops;
  std::unordered_set<size_t> after_deps;

  for (size_t pos : *cur_it) {
    if (cur_ops.empty()) {
      cur_ops.emplace(pos);
      continue;
    }
    auto *op_handle = all_ops[pos].get();
    if (IsAnyOfParentInCurrentOp(op_handle, cur_ops, offset_map)) {
      after_deps.emplace(pos);
    } else if (IsAnyOfChildInCurrentOp(op_handle, cur_ops, offset_map)) {
      before_deps.emplace(pos);
    } else {
      cur_ops.emplace(pos);
    }
  }

  if (!before_deps.empty()) {
    ResolveOpDeps(all_ops, offset_map, res->insert(cur_it, before_deps), res);
  }

  *cur_it = cur_ops;

  if (!after_deps.empty()) {
    ++cur_it;
    ResolveOpDeps(all_ops, offset_map, res->insert(cur_it, after_deps), res);
  }
}

std::vector<FuseAllReduceGraphBuilder::NCCLAllReduceGroup>
FuseAllReduceGraphBuilder::GetNotDependedAllReduceOp(
    SSAGraph *graph, const BlockDesc &global_block) const {
  std::unordered_map<std::type_index,
                     std::vector<std::unique_ptr<OpHandleBase>>>
      all_reduce_ops;

  for (size_t i = 0; i < graph->ops_.size();) {
    if (dynamic_cast<NCCLAllReduceOpHandle *>(graph->ops_[i].get())) {
      // The op.size() will shrink after extract op, so ++i is not needed.
      auto op = graph->ExtractOp(i);
      std::type_index data_type = typeid(void);
      for (auto &ipt : op->Inputs()) {
        auto *input_var_handle = dynamic_cast<VarHandle *>(ipt);
        if (input_var_handle) {
          data_type = ToTypeIndex(
              global_block.FindVar(input_var_handle->name_)->GetDataType());
          break;
        }
      }
      PADDLE_ENFORCE(data_type != typeid(void));
      all_reduce_ops[data_type].emplace_back(std::move(op));
    } else {
      ++i;
    }
  }

  if (all_reduce_ops.empty()) {
    return std::vector<FuseAllReduceGraphBuilder::NCCLAllReduceGroup>();
  }

  std::vector<NCCLAllReduceGroup> res_vec;

  for (auto &all_reduce_op_with_type : all_reduce_ops) {
    auto &all_reduce_op = all_reduce_op_with_type.second;
    std::unordered_map<OpHandleBase *, size_t> offsets;
    std::list<std::unordered_set<size_t>> res;
    res.emplace_back();
    for (size_t i = 0; i < all_reduce_op.size(); ++i) {
      offsets.emplace(all_reduce_op[i].get(), i);
      res.back().emplace(i);
    }

    ResolveOpDeps(all_reduce_op, offsets, res.begin(), &res);

    for (auto &set : res) {
      NCCLAllReduceGroup group;
      group.type_ = all_reduce_op_with_type.first;

      for (auto pos : set) {
        group.ops_.emplace(std::move(all_reduce_op[pos]));
      }
      res_vec.emplace_back(std::move(group));
    }
  }

  VLOG(5) << "There are " << res_vec.size() << " group of reduce_all";
  if (VLOG_IS_ON(10)) {
    for (size_t i = 0; i < res_vec.size(); ++i) {
      std::ostringstream sout;
      sout << "Group " << i << " with type " << res_vec[i].type_.name() << "\n";

      for (const std::unique_ptr<OpHandleBase> &op : res_vec[i].ops_) {
        sout << "\t" << op->DebugString() << "\n";
      }
      VLOG(10) << sout.str();
    }
  }

  return res_vec;
}

struct FuseVarParams {
  std::string var_name;
  size_t version;
};

static VarHandle *FuseVariable(SSAGraph *graph, size_t scope_idx, Scope *scope,
                               platform::Place place,
                               const std::string &whole_varaible_name,
                               const std::vector<FuseVarParams> &fused_vars,
                               const BlockDesc &global_block,
                               std::type_index type) {
  if (VLOG_IS_ON(10)) {
    VLOG(10) << "Fusing variable:";
    for (auto &var_param : fused_vars) {
      VLOG(10) << "\t" << var_param.var_name << ", " << var_param.version;
    }
  }
  std::unordered_map<std::string, int64_t> inputs_numel;
  for (auto fused_var : fused_vars) {
    auto fused_var_name = fused_var.var_name;
    auto *var_desc = global_block.FindVar(fused_var_name);
    PADDLE_ENFORCE_NOT_NULL(var_desc);
    int64_t numel =
        framework::product(framework::make_ddim(var_desc->GetShape()));
    PADDLE_ENFORCE_GT(numel, 0);
    inputs_numel[fused_var_name] = numel;
  }
  PADDLE_ENFORCE_EQ(inputs_numel.size(), fused_vars.size());

  auto *fuse_vars_op_handle =
      new FuseVarsOpHandle(scope, place, inputs_numel, type);
  graph->ops_.emplace_back(fuse_vars_op_handle);

  // CreateFuseVarsOpHandleIO
  fuse_vars_op_handle->SetDeviceContext(
      place, platform::DeviceContextPool::Instance().Get(place));

  auto out_var_handle =
      graph->AppendVariable(whole_varaible_name, scope_idx, place);
  fuse_vars_op_handle->AddOutput(out_var_handle);

  for (size_t j = 0; j < fused_vars.size(); ++j) {
    auto o_handle = graph->InsertVariable(
        fused_vars.at(j).version, fused_vars.at(j).var_name, scope_idx, place);
    fuse_vars_op_handle->AddOutput(o_handle);
  }

  // Add dependence
  for (size_t i = 0; i < fused_vars.size(); ++i) {
    size_t position = fused_vars.at(i).version;
    auto &version_vars =
        graph->vars_.at(scope_idx).at(fused_vars.at(i).var_name);
    if (position > 0) {
      for (auto op : version_vars.at(position - 1).get()->pending_ops_) {
        auto *dep_var = new DummyVarHandle();
        op->AddOutput(dep_var);
        fuse_vars_op_handle->AddInput(dep_var);
        graph->dep_vars_.emplace(dep_var);
      }
    }
    if (position < version_vars.size() - 1) {
      auto &op = version_vars.at(position + 1).get()->generated_op_;
      auto *dep_var = new DummyVarHandle();
      op->AddInput(dep_var);
      fuse_vars_op_handle->AddOutput(dep_var);
      graph->dep_vars_.emplace(dep_var);
    }
  }

  return out_var_handle;
}

static int UniqID() {
  static int id = 0;
  return id++;
}

static VarHandle *FindTheFirstVersionOfVariable(VarHandle *var) {
  std::unordered_set<OpHandleBase *> visited;
  VarHandle *result = var;

  std::function<void(VarHandle *)> impl = [&](VarHandle *cur_var) {
    if (cur_var->name_ == result->name_ &&
        cur_var->scope_idx_ == result->scope_idx_ &&
        cur_var->version_ < result->version_) {
      result = cur_var;
    }

    if (cur_var->generated_op_ == nullptr ||
        visited.count(cur_var->generated_op_)) {
      return;
    }
    visited.emplace(cur_var->generated_op_);
    for (auto *input : cur_var->generated_op_->Inputs()) {
      if (auto *prev_var = dynamic_cast<VarHandle *>(input)) {
        impl(prev_var);
      }
    }
  };
  impl(result);
  return result;
}

void FuseAllReduceGraphBuilder::FuseAllReduceOp(
    SSAGraph *graph, NCCLAllReduceGroup &&group,
    const BlockDesc &global_block) const {
  auto &ops = group.ops_;
  if (ops.empty()) {
    return;
  } else if (ops.size() == 1) {
    auto it = ops.begin();
    graph->ops_.emplace_back(
        const_cast<std::unique_ptr<OpHandleBase> &>(*it).release());
    return;
  }

  // 1.Insert Fuse Op
  // For each place insert one fuse var op
  std::vector<VarHandle *> fused_vars = GetFusedGradient(
      graph, string::Sprintf("all_grad_%d", UniqID()), global_block, group);

  // 2.Insert All reduce to fused var.
  OpHandleBase *nccl_op_handle = nullptr;
  {
    // 1. Create Output.
    auto name = fused_vars[0]->name_;
    std::vector<VarHandle *> output_fused_var;
    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      auto &var_map = graph->vars_.at(i);
      auto &var_list = var_map.at(name);
      VarHandle *var_handle =
          new VarHandle(var_list.size(), i, name, places_.at(i));
      var_list.emplace_back(var_handle);
      output_fused_var.emplace_back(var_handle);
    }

    // 2. Create All Reduce Op
    auto *op_handle = new NCCLAllReduceOpHandle(local_scopes_, places_, *ctxs_);

    for (auto *ipt : fused_vars) {
      op_handle->AddInput(ipt);
    }

    for (auto *opt : output_fused_var) {
      op_handle->AddOutput(opt);
    }
    graph->ops_.emplace_back(op_handle);
    nccl_op_handle = op_handle;
  }

  // 3. Replace all reduce op
  {
    for (const std::unique_ptr<OpHandleBase> &nccl_op : group.ops_) {
      auto *to_remove_op =
          reinterpret_cast<NCCLAllReduceOpHandle *>(nccl_op.get());

      auto &inputs = nccl_op->Inputs();
      auto &outputs = nccl_op->Outputs();

      for (size_t i = 0, j = 0; i < inputs.size() && j < outputs.size();
           ++i, ++j) {
        VarHandle *in = nullptr;
        while ((in = dynamic_cast<VarHandle *>(inputs[i])) == nullptr) {
          // in is a dummy variable. Move this dummy variable to
          auto *dummy_in = inputs[i];
          dummy_in->pending_ops_.erase(
              dummy_in->pending_ops_.find(to_remove_op));
          nccl_op_handle->AddInput(dummy_in);
          inputs[i] = nullptr;  // drop inputs.
          ++i;
        }
        VarHandle *out = nullptr;
        while ((out = dynamic_cast<VarHandle *>(outputs[j])) == nullptr) {
          // out is a dummy variable. Move this dummy variable to all_reduce
          auto *dummy_out = outputs[j];
          nccl_op_handle->AddOutput(dummy_out);
          outputs[j] = nullptr;
          ++j;
        }

        PADDLE_ENFORCE_EQ(in->scope_idx_, out->scope_idx_);

        // Here, in & out are paired.

        // Link out's pending_ops to in and AllReduceOp
        auto pending_ops = out->pending_ops_;

        // pending_ops are SGD like operators.
        for (OpHandleBase *sgd_op : pending_ops) {
          // Replace out to in
          sgd_op->ReplaceInput(out, in);

          // Link nccl_op to pending_ops
          auto *dummy = new DummyVarHandle();
          nccl_op_handle->AddOutput(dummy);
          sgd_op->AddInput(dummy);
          graph->dep_vars_.emplace(dummy);
        }
        PADDLE_ENFORCE(out->pending_ops_.empty());

        if (in->generated_op_) {
          auto *dummy = new DummyVarHandle();
          in->generated_op_->AddOutput(dummy);
          nccl_op_handle->AddInput(dummy);
          graph->dep_vars_.emplace(dummy);
        }

        // Remove out variable, which is not used.
        graph->ExtractVariable(out->version_, out->name_, out->scope_idx_);
      }
    }
  }
}
std::vector<VarHandle *> FuseAllReduceGraphBuilder::GetFusedGradient(
    SSAGraph *graph, const std::string &fused_var_name,
    const BlockDesc &global_block, const NCCLAllReduceGroup &group) const {
  std::vector<VarHandle *> fuse_params;

  for (auto &op : group.ops_) {
    for (VarHandleBase *var : op->Inputs()) {
      if (auto *var_handle = dynamic_cast<VarHandle *>(var)) {
        // NOTE: Here we assume all variables on each device are same version.
        // It is true for all_reduce strategy.
        fuse_params.emplace_back(var_handle);
        break;
      }
    }
  }

  for (auto &var : fuse_params) {
    if (var->version_ == 0) continue;
    // Find the first version of the variable.
    var = FindTheFirstVersionOfVariable(var);
  }

  std::vector<FuseVarParams> params;
  for (auto &var : fuse_params) {
    params.emplace_back(FuseVarParams{var->name_, var->version_});
  }

  std::vector<VarHandle *> fused_vars;

  for (size_t i = 0; i < places_.size(); ++i) {
    fused_vars.emplace_back(FuseVariable(graph, i, local_scopes_[i], places_[i],
                                         fused_var_name, params, global_block,
                                         group.type_));
  }
  return fused_vars;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
