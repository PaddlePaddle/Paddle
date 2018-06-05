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
  // TODO(yy): Complete this method.
  auto graph = builder_->Build(program);

  auto all_reduce_ops =
      GetNotDependedAllReduceOp(graph.get(), program.Block(0));

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
  // TODO(zcd): Complete this method
  return nullptr;
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
  std::vector<VarHandle *> fused_vars =
      GetFusedGradient(graph, global_block, group);

  // 2.Insert All reduce to fused var.

  // 3. Remove all reduce op

  // 4. Correct handle deps

  //
  //
  //
  //  // Get input and output
  //  std::vector<std::vector<VarHandle *>> inputs;
  //  std::vector<std::vector<VarHandle *>> outputs;
  //
  //  std::vector<std::vector<VarHandle *>> dep_inputs;
  //  std::vector<std::vector<VarHandle *>> dep_outputs;
  //
  //  inputs.resize(ops.begin()->get()->Inputs().size());
  //  outputs.resize(ops.begin()->get()->Inputs().size());
  //
  //  // Collect AllReduce Ops input and output
  //  auto collect_input_output = [](
  //      VarHandleBase *var_handle_base,
  //      std::vector<std::vector<VarHandle *>> *vec,
  //      std::vector<std::vector<VarHandle *>> *dep_vec) {
  //    auto var_h = dynamic_cast<VarHandle *>(var_handle_base);
  //    if (var_h) {
  //      int dev_id = boost::get<platform::CUDAPlace>(var_h->place_).device;
  //      (*vec)[dev_id].emplace_back(var_h);
  //    } else {
  //      // auto dummy_h = dynamic_cast<DummyVarHandle *>(var_handle_base);
  //      // pass
  //    }
  //  };
  //  for (auto op_iter = ops.begin(); op_iter != ops.end(); op_iter++) {
  //    auto op = op_iter->get();
  //    for (size_t i = 0; i < op->Inputs().size(); ++i) {
  //      collect_input_output(op->Inputs()[i], &inputs, &dep_inputs);
  //      collect_input_output(op->Outputs()[i], &outputs, &dep_outputs);
  //    }
  //  }
  //
  //  // Create FuseVarsOpHandles
  //  std::vector<std::string> fused_vars_output_names;
  //  std::vector<OpHandleBase *> fuse_vars_ops;
  //  for (size_t dev_id = 0; dev_id < inputs.size(); ++dev_id) {
  //    PADDLE_ENFORCE_EQ(inputs[dev_id].size(), outputs[dev_id].size());
  //    auto *scope = local_scopes_[dev_id];
  //    auto &place = places_[dev_id];
  //
  //    auto fused_vars_output_name =
  //        string::Sprintf("FuseVars_GPU%d_%d", dev_id, GetUniqNum());
  //    fused_vars_output_names.emplace_back(fused_vars_output_name);
  //
  //    std::unordered_map<std::string, int64_t> inputs_dims;
  //    proto::VarType::Type var_type =
  //        global_block.FindVar(inputs[dev_id][0]->name_)->GetDataType();
  //    PADDLE_ENFORCE_EQ(ToDataType(group.type_), var_type);
  //    for (size_t j = 0; j < inputs[dev_id].size(); ++j) {
  //      auto var_desc = global_block.FindVar(inputs[dev_id][j]->name_);
  //      inputs_dims[inputs[dev_id][j]->name_] =
  //          framework::product(framework::make_ddim(var_desc->GetShape()));
  //      PADDLE_ENFORCE_EQ(var_desc->GetDataType(), var_type);
  //    }
  //
  //    auto *op_handle =
  //        new FuseVarsOpHandle(scope, place, inputs_dims, group.type_);
  //    fuse_vars_ops.emplace_back(op_handle);
  //    CreateFuseVarsOpHandleIO(graph, op_handle, dev_id,
  //    fused_vars_output_name,
  //                             place, inputs[dev_id]);
  //  }
  //
  //  // Insert fuse vars into graph
  //  InsertFusedVarsOpHandleIntoGraph(graph, &inputs, fuse_vars_ops);
  //
  //  // Create fused_nccl_all_reduce op handle
  //  auto *nccl_op_handle =
  //      new NCCLAllReduceOpHandle(local_scopes_, places_, *ctxs_);
  //  graph->ops_.emplace_back(nccl_op_handle);
  //
  //  CreateNCCLAllReduceOpHandleIO(fused_vars_output_names, &inputs, &outputs,
  //                                nccl_op_handle, graph);
}
std::vector<VarHandle *> FuseAllReduceGraphBuilder::GetFusedGradient(
    SSAGraph *graph, const BlockDesc &global_block,
    const NCCLAllReduceGroup &group) const {
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
    fused_vars.emplace_back(
        FuseVariable(graph, i, local_scopes_[i], places_[i],
                     string::Sprintf("all_grad_%d", UniqID()), params,
                     global_block, group.type_));
  }
  return fused_vars;
}

void FuseAllReduceGraphBuilder::CreateNCCLAllReduceOpHandleIO(
    const std::vector<std::string> &fused_var_names,
    std::vector<std::vector<VarHandle *>> *inputs,
    std::vector<std::vector<VarHandle *>> *outputs,
    NCCLAllReduceOpHandle *nccl_op_handle, SSAGraph *graph) const {
  // Add inputs
  auto &ins = *inputs;
  for (size_t dev_id = 0; dev_id < ins.size(); ++dev_id) {
    auto &fused_vars = graph->vars_[dev_id][fused_var_names[dev_id]];
    nccl_op_handle->AddInput(fused_vars.rbegin()->get());

    // Add dependence
    auto *dep_var = new DummyVarHandle();
    nccl_op_handle->AddInput(dep_var);
    for (size_t j = 0; j < ins[dev_id].size(); ++j) {
      ins[dev_id][j]->generated_op_->AddOutput(dep_var);
    }
  }

  // Add outputs
  auto &outs = *outputs;
  for (size_t dev_id = 0; dev_id < outs.size(); ++dev_id) {
    for (size_t j = 0; j < outs[dev_id].size(); ++j) {
      nccl_op_handle->AddOutput(outs[dev_id][j]);
    }
  }
}

void FuseAllReduceGraphBuilder::InsertFusedVarsOpHandleIntoGraph(
    SSAGraph *graph, std::vector<std::vector<VarHandle *>> *inputs,
    const std::vector<OpHandleBase *> &fuse_vars_ops) const {
  auto &ins = *inputs;
  for (size_t dev_id = 0; dev_id < ins.size(); ++dev_id) {
    // Insert fuse_vars_op to graph
    graph->ops_.emplace_back(fuse_vars_ops[dev_id]);
    // Add dependence
    for (auto in : ins[dev_id]) {
      if (in->generated_op_) {
        in->generated_op_->AddInput(fuse_vars_ops[dev_id]->Outputs()[0]);
      }
    }
  }
}

void FuseAllReduceGraphBuilder::CreateFuseVarsOpHandleIO(
    SSAGraph *graph, OpHandleBase *op_handle, const int dev_id,
    const std::string fused_var_name, const platform::Place &place,
    const std::vector<VarHandle *> &inputs) const {
  op_handle->SetDeviceContext(
      place, platform::DeviceContextPool::Instance().Get(place));

  // Add input
  for (size_t j = 0; j < inputs.size(); ++j) {
    auto var_name = inputs[j]->name_;
    auto &vars = graph->vars_[dev_id][var_name];
    std::unique_ptr<VarHandle> in_var(
        new VarHandle(0, dev_id, var_name, place));
    vars.insert(vars.begin(), std::move(in_var));

    for (size_t k = 1; k < vars.size(); ++k) {
      vars[k]->version_++;
    }
    op_handle->AddInput(vars.begin()->get());
  }

  // Add output
  auto &vars = graph->vars_[dev_id][fused_var_name];
  size_t version = vars.size();
  auto out_var = new VarHandle(version, dev_id, fused_var_name, place);
  vars.emplace_back(out_var);
  op_handle->AddOutput(out_var);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
