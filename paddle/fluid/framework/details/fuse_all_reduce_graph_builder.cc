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

  std::unordered_map<std::string, VarDesc *> all_vars_desc;
  for (auto *var : program.Block(0).AllVars()) {
    all_vars_desc[var->Name()] = var;
  }

  auto all_reduce_ops = GetNotDependedAllReduceOp(graph.get());

  for (auto &op_group : all_reduce_ops) {
    FuseAllReduceOp(graph.get(), std::move(op_group), all_vars_desc);
  }
  return graph;
}

inline static bool IsParentOpInCurrentOp(
    OpHandleBase *op, const std::unordered_set<size_t> &cur_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map) {
  if (cur_ops.count(offset_map.at(op))) {
    return true;
  }

  for (auto *in : op->Inputs()) {
    if (in->generated_op_ &&
        IsParentOpInCurrentOp(in->generated_op_, cur_ops, offset_map)) {
      return true;
    }
  }

  return false;
}

inline static bool IsChildOpInCurrentOp(
    OpHandleBase *op, const std::unordered_set<size_t> &cur_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map) {
  if (cur_ops.count(offset_map.at(op))) {
    return true;
  }

  for (auto *out : op->Outputs()) {
    for (auto *out_op : out->pending_ops_) {
      if (IsChildOpInCurrentOp(out_op, cur_ops, offset_map)) {
        return true;
      }
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
  std::unordered_set<size_t> after_deps;
  std::unordered_set<size_t> cur_ops;

  for (size_t pos : *cur_it) {
    if (cur_ops.empty()) {
      cur_ops.emplace(pos);
      continue;
    }
    auto *op_handle = all_ops[pos].get();

    if (IsParentOpInCurrentOp(op_handle, cur_ops, offset_map)) {
      after_deps.emplace(pos);
    } else if (IsChildOpInCurrentOp(op_handle, cur_ops, offset_map)) {
      before_deps.emplace(pos);
    } else {
      cur_ops.emplace(pos);
    }
  }

  if (!before_deps.empty()) {
    ResolveOpDeps(all_ops, offset_map, res->insert(cur_it, before_deps), res);
  }

  cur_it->swap(cur_ops);

  if (!after_deps.empty()) {
    ++cur_it;
    ResolveOpDeps(all_ops, offset_map, res->insert(cur_it, after_deps), res);
  }
}

std::vector<std::unordered_set<std::unique_ptr<OpHandleBase>>>
FuseAllReduceGraphBuilder::GetNotDependedAllReduceOp(SSAGraph *graph) const {
  std::vector<std::unique_ptr<OpHandleBase>> all_reduce_ops;

  for (size_t i = 0; i < graph->ops_.size();) {
    if (dynamic_cast<NCCLAllReduceOpHandle *>(graph->ops_[i].get())) {
      // The op.size() will shrink after extract op, so ++i is not needed.
      all_reduce_ops.emplace_back(graph->ExtractOp(i));
    } else {
      ++i;
    }
  }
  std::unordered_map<OpHandleBase *, size_t> offsets;
  std::list<std::unordered_set<size_t>> res;
  res.emplace_back();
  for (size_t i = 0; i < all_reduce_ops.size(); ++i) {
    offsets.emplace(all_reduce_ops[i].get(), i);
    res.back().emplace(i);
  }

  ResolveOpDeps(all_reduce_ops, offsets, res.begin(), &res);

  std::vector<std::unordered_set<std::unique_ptr<OpHandleBase>>> res_vec;
  for (auto &set : res) {
    res_vec.emplace_back();
    auto &pointer_set = res_vec.back();

    for (auto pos : set) {
      pointer_set.emplace(std::move(all_reduce_ops[pos]));
    }
  }

  if (VLOG_IS_ON(10)) {
    for (size_t i = 0; i < res_vec.size(); ++i) {
      std::ostringstream sout;
      sout << "Group " << i << "\n";

      for (const std::unique_ptr<OpHandleBase> &op : res_vec[i]) {
        sout << "\t" << op->DebugString() << "\n";
      }
      VLOG(10) << sout.str();
    }
  }

  return res_vec;
}

void FuseAllReduceGraphBuilder::FuseAllReduceOp(
    SSAGraph *graph, std::unordered_set<std::unique_ptr<OpHandleBase>> &&ops,
    const std::unordered_map<std::string, VarDesc *> &all_vars_desc) const {
  if (ops.size() <= 1) return;
  // Get input and output
  std::vector<std::vector<VarHandle *>> inputs;
  std::vector<std::vector<VarHandle *>> outputs;
  inputs.resize(ops.begin()->get()->Inputs().size());
  outputs.resize(ops.begin()->get()->Inputs().size());

  // Collect AllReduce Ops input and output
  auto collect_input_output = [](VarHandleBase *var_handle_base,
                                 std::vector<std::vector<VarHandle *>> *vec) {
    auto var_h = dynamic_cast<VarHandle *>(var_handle_base);
    if (var_h) {
      int dev_id = boost::get<platform::CUDAPlace>(var_h->place_).device;
      (*vec)[dev_id].emplace_back(var_h);
    }
  };
  for (auto op_iter = ops.begin(); op_iter != ops.end(); op_iter++) {
    auto op = op_iter->get();
    for (size_t i = 0; i < op->Inputs().size(); ++i) {
      collect_input_output(op->Inputs()[i], &inputs);
      collect_input_output(op->Outputs()[i], &outputs);
    }
  }

  // Create FuseVarsOpHandles
  std::vector<std::string> fused_vars_output_names;
  std::vector<OpHandleBase *> fuse_vars_ops;
  for (size_t dev_id = 0; dev_id < inputs.size(); ++dev_id) {
    PADDLE_ENFORCE_EQ(inputs[dev_id].size(), outputs[dev_id].size());
    auto *scope = local_scopes_[dev_id];
    auto &place = places_[dev_id];

    auto fused_vars_output_name =
        string::Sprintf("FuseVars_GPU%d_%d", dev_id, GetUniqNum());
    fused_vars_output_names.emplace_back(fused_vars_output_name);

    std::unordered_map<std::string, int64_t> inputs_dims;
    proto::VarType::Type var_type =
        all_vars_desc.at(inputs[dev_id][0]->name_)->GetType();

    for (size_t j = 0; j < inputs[dev_id].size(); ++j) {
      auto var_desc = all_vars_desc.at(inputs[dev_id][j]->name_);
      inputs_dims[inputs[dev_id][j]->name_] =
          framework::product(framework::make_ddim(var_desc->GetShape()));
      PADDLE_ENFORCE_EQ(var_desc->GetType(), var_type);
    }

    auto *op_handle = new FuseVarsOpHandle(scope, place, inputs_dims, var_type);
    fuse_vars_ops.emplace_back(op_handle);
    CreateFuseVarsOpHandleIO(graph, op_handle, dev_id, fused_vars_output_name,
                             place, inputs[dev_id]);
  }

  // Insert fuse vars into graph
  InsertFusedVarsOpHandleToGraph(graph, &inputs, fuse_vars_ops);

  // Create fused_nccl_all_reduce op handle
  auto *nccl_op_handle =
      new NCCLAllReduceOpHandle(local_scopes_, places_, *ctxs_);
  graph->ops_.emplace_back(nccl_op_handle);

  CreateNCCLAllReduceOpHandleIO(fused_vars_output_names, &inputs, &outputs,
                                nccl_op_handle, graph);
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

void FuseAllReduceGraphBuilder::InsertFusedVarsOpHandleToGraph(
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
