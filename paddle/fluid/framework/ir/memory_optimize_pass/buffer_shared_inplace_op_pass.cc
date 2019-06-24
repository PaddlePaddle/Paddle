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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
struct InplaceVarPair {
  InplaceVarPair(const std::string &in_param, details::VarHandle *in_var,
                 const std::string &out_param, details::VarHandle *out_var)
      : in_param_(in_param),
        in_var_(in_var),
        out_param_(out_param),
        out_var_(out_var) {}

  std::string in_param_;
  details::VarHandle *in_var_;
  std::string out_param_;
  details::VarHandle *out_var_;
};

class BufferSharedInplaceOpPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;

 private:
  static std::unordered_set<ir::Node *> FindNodesByName(
      const std::vector<ir::Node *> &nodes, const std::string &name);

  static std::unordered_multiset<std::string> GetAllInputArgs(
      const framework::OpDesc *op_desc);

  static ir::Node *FindOccuredOnceInputNode(ir::Node *op_node,
                                            const std::string &in_arg);

  static VarDesc *GetVarDescOfVar(const details::GraphVars &vars,
                                  size_t scope_idx,
                                  const std::string &var_name);

  static void InsertShareTensorBufferOpHandleToGraph(
      ir::Graph *graph, details::GraphVars *all_vars,
      const ir::MemOptVarInfoMapList &mem_opt_var_infos,
      details::ComputationOpHandle *op,
      const std::vector<InplaceVarPair> &inplace_vars);
};

std::unordered_set<ir::Node *> BufferSharedInplaceOpPass::FindNodesByName(
    const std::vector<ir::Node *> &nodes, const std::string &name) {
  std::unordered_set<ir::Node *> result;
  for (auto *node : nodes) {
    if (node->Name() == name) {
      result.insert(node);
    }
  }
  return result;
}

std::unordered_multiset<std::string> BufferSharedInplaceOpPass::GetAllInputArgs(
    const framework::OpDesc *op_desc) {
  std::unordered_multiset<std::string> result;
  for (auto &pair : op_desc->Inputs()) {
    for (auto &arg_name : pair.second) {
      result.insert(arg_name);
    }
  }
  return result;
}

ir::Node *BufferSharedInplaceOpPass::FindOccuredOnceInputNode(
    ir::Node *op_node, const std::string &in_arg) {
  auto in_nodes = FindNodesByName(op_node->inputs, in_arg);
  if (in_nodes.size() != 1) {
    return nullptr;
  }

  if (GetAllInputArgs(op_node->Op()).count(in_arg) != 1) {
    return nullptr;
  }

  if (!FindNodesByName(op_node->outputs, in_arg).empty()) {
    return nullptr;
  }

  return *(in_nodes.begin());
}

VarDesc *BufferSharedInplaceOpPass::GetVarDescOfVar(
    const details::GraphVars &vars, size_t scope_idx,
    const std::string &var_name) {
  auto iter = vars[scope_idx].find(var_name);
  PADDLE_ENFORCE(iter != vars[scope_idx].end(), "Cannot find variable %s",
                 var_name);
  auto *var_desc = TryGetLatestVarDesc(iter->second);
  PADDLE_ENFORCE_NOT_NULL(var_desc, "VarDesc of variable %s cannot be found",
                          var_name);
  return var_desc;
}

void BufferSharedInplaceOpPass::ApplyImpl(ir::Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  bool use_cuda = Get<bool>(kUseCuda);
  auto &all_vars = graph->Get<details::GraphVars>(details::kGraphVars);

  // Step 1: Build a reverse map of last_live_ops
  // i.e.: op -> vars
  std::unordered_map<details::ComputationOpHandle *,
                     std::unordered_map<std::string, ir::Node *>>
      candidate_ops;
  for (auto &each_scope_ops : last_live_ops) {
    for (auto &pair : each_scope_ops) {
      // If variable has more than 1 last lived ops, this variable cannot
      // be inplaced.
      if (pair.second.size() != 1) {
        continue;
      }

      auto *op = *(pair.second.begin());
      const std::string &op_type = op->GetOp()->Type();
      const framework::OpDesc *op_desc = op->Node()->Op();
      PADDLE_ENFORCE_NOT_NULL(op_desc);

      auto &infer_inplace = OpInfoMap::Instance().Get(op_type).infer_inplace_;
      if (!infer_inplace) {
        continue;
      }

      const std::string &var_name = pair.first;
      ir::Node *in_node = FindOccuredOnceInputNode(op->Node(), var_name);
      if (in_node) {
        candidate_ops[op][var_name] = in_node;
      }
    }
  }

  // Step 2: Check which vars can be inplaced indeed
  std::unordered_map<details::ComputationOpHandle *,
                     std::vector<InplaceVarPair>>
      inplace_ops;
  for (auto &op_vars_pair : candidate_ops) {
    auto *op = op_vars_pair.first;
    auto &vars = op_vars_pair.second;
    size_t scope_idx = op->GetScopeIdx();

    const std::string &op_type = op->GetOp()->Type();
    auto *op_desc = op->Node()->Op();

    auto in_to_outs =
        OpInfoMap::Instance().Get(op_type).infer_inplace_(*op_desc, use_cuda);
    for (auto &pair : in_to_outs) {
      auto &in_param = pair.first;
      auto &in_args = op_desc->Input(in_param);
      if (in_args.empty()) {
        VLOG(4) << "Cannot inplace because Input(" << in_param
                << ") is empty in " << op_type;
        continue;
      }

      auto &in_arg = in_args[0];
      auto iter = vars.find(in_arg);
      if (iter == vars.end()) {
        VLOG(4) << "Cannot inplace maybe because Input(" << in_param
                << ")=" << in_arg << " is not lastly used in op " << op_type
                << ", or it occurs multiple times in input or occurs in output";
        continue;
      }

      ir::Node *in_node = iter->second;

      auto &out_param = pair.second;
      auto &out_args = op_desc->Output(out_param);

      if (out_args.empty()) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ") is empty in " << op_type;
        continue;
      }

      auto &out_arg = out_args[0];
      if (in_arg == out_arg) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is the same with Output(" << out_param << ")=" << out_arg
                << " in " << op_type;
        continue;
      }

      if (!FindNodesByName(op->Node()->inputs, out_arg).empty()) {
        VLOG(4) << "Cannot inplace because Output(" << in_param
                << ")=" << out_arg << " occurs in input of op " << op_type;
        continue;
      }

      // Check whether input is persistable
      // And check input/output are all LoDTensors
      // TODO(zjl): support SelectedRows inplace
      auto *in_var_desc = GetVarDescOfVar(all_vars, scope_idx, in_arg);
      if (in_var_desc->Persistable()) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is persistable";
        continue;
      }

      auto *out_var_desc = GetVarDescOfVar(all_vars, scope_idx, out_arg);
      if (out_var_desc->Persistable()) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ")=" << out_arg << " is persistable";
        continue;
      }

      if (in_var_desc->Proto()->type().type() != proto::VarType::LOD_TENSOR ||
          out_var_desc->Proto()->type().type() != proto::VarType::LOD_TENSOR) {
        VLOG(4) << "Cannot inplace because not LOD_TENSOR";
        continue;
      }

      PADDLE_ENFORCE(all_vars[scope_idx].count(out_arg) > 0,
                     "Variable %s is not found in ssa graph", out_arg);
      auto &out_var_handles = all_vars[scope_idx].at(out_arg);

      auto out_nodes = FindNodesByName(op->Node()->outputs, out_arg);
      PADDLE_ENFORCE_EQ(out_nodes.size(), 1, "Output(%s)=%s must be unique",
                        out_param, out_arg);
      ir::Node *out_node = *(out_nodes.begin());

      auto var_handle_iter =
          std::find_if(out_var_handles.begin(), out_var_handles.end(),
                       [out_node](details::VarHandle *handle) {
                         return handle->Node() == out_node;
                       });

      PADDLE_ENFORCE(var_handle_iter != out_var_handles.end(),
                     "Variable %s is not found in VarHandles of ssa graph",
                     out_arg);

      details::VarHandle *out_var_handle = *var_handle_iter;
      auto &in_var_handle_base = in_node->Wrapper<details::VarHandleBase>();
      auto *in_var_handle =
          dynamic_cast<details::VarHandle *>(&in_var_handle_base);
      PADDLE_ENFORCE_NOT_NULL(in_var_handle);
      inplace_ops[op].emplace_back(in_param, in_var_handle, out_param,
                                   out_var_handle);
    }
  }

  // Step 3: Insert new VarHandle and new OpHandle to the graph
  const auto &mem_opt_var_info =
      Get<ir::MemOptVarInfoMapList>(kMemOptVarInfoMapList);
  for (auto &pair : inplace_ops) {
    InsertShareTensorBufferOpHandleToGraph(graph, &all_vars, mem_opt_var_info,
                                           pair.first, pair.second);
  }
}
*/

/**
 * Suppose op has n inputs: i_1, i_2, ... , i_n,
 * and n outputs: o_1, o_2, ... , o_n, where o_j can be inplaced with i_j.
 *
 * We insert a ShareTensorBufferOpHandle inplace_op to perform inplace
 * during runtime.
 *
 * The dependency should be:
 *
 * 1. inplace_op takes all inputs of op as its input (not only
 *    i_1, i_2, ... , i_n), and takes o_1, o_2, ... , o_n as its
 *    outputs. Notice that outputs of inplace_op should be new vars
 *    of o_1, o_2, ..., o_n, whose versions are 1 less thaa_
 *    outputs of op.
 *
 * 2. inplace_op takes a dep_var as its output, and op takes the same
 *    dep_var as its input. This is designed to resolve write-after-write
 *    data hazard between inplace_op and op.
 */
/*
void BufferSharedInplaceOpPass::InsertShareTensorBufferOpHandleToGraph(
    ir::Graph *graph, details::GraphVars *all_vars,
    const ir::MemOptVarInfoMapList &mem_opt_var_infos,
    details::ComputationOpHandle *op,
    const std::vector<InplaceVarPair> &inplace_vars) {
  auto idx = op->GetScopeIdx();
  auto *buffer_share_node =
      graph->CreateEmptyNode("inplace", ir::Node::Type::kOperation);

  std::vector<ir::MemOptVarInfo *> in_vars;
  std::vector<std::string> out_vars;
  std::vector<std::pair<std::string, std::string>> in_out_params;
  in_vars.reserve(inplace_vars.size());
  out_vars.reserve(inplace_vars.size());
  in_out_params.reserve(inplace_vars.size());
  for (auto &inplace_var : inplace_vars) {
    PADDLE_ENFORCE(
        mem_opt_var_infos[idx].count(inplace_var.in_var_->Name()) > 0,
        "Cannot find in var %s", inplace_var.in_var_->Name());
    in_vars.emplace_back(
        mem_opt_var_infos[idx].at(inplace_var.in_var_->Name()).get());
    out_vars.emplace_back(inplace_var.out_var_->Name());
    in_out_params.emplace_back(inplace_var.in_param_, inplace_var.out_param_);
  }

  auto *buffer_share_op = new details::InplaceShareTensorBufferOpHandle(
      buffer_share_node, op->GetScope(), op->GetScopeIdx(), op->GetOp()->Type(),
      in_out_params, in_vars, out_vars);

  for (auto &inplace_var : inplace_vars) {
    auto *out_var = inplace_var.out_var_;
    auto *new_out_var_node = graph->CreateVarNode(out_var->Node()->Var());
    auto *new_out_var = new details::VarHandle(
        new_out_var_node, out_var->version(), out_var->scope_idx(),
        out_var->Name(), out_var->place());

    auto &all_version_out_var = (*all_vars)[idx].at(out_var->Name());
    auto iter = std::find(all_version_out_var.begin(),
                          all_version_out_var.end(), out_var);
    PADDLE_ENFORCE(iter != all_version_out_var.end());

    iter = all_version_out_var.insert(iter, new_out_var);

    while (++iter != all_version_out_var.end()) {
      // increase version index for the following vars
      (*iter)->set_version((*iter)->version() + 1);
    }

    buffer_share_op->AddOutput(new_out_var);
  }

  for (auto *in_var : op->Inputs()) {
    buffer_share_op->AddInput(in_var);
  }

  auto *dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
  graph->Get<details::GraphDepVars>(details::kGraphDepVars).emplace(dep_var);
  op->AddInput(dep_var);
  buffer_share_op->AddOutput(dep_var);

  buffer_share_op->SetDeviceContext(
      op->GetPlace(),
      platform::DeviceContextPool::Instance().Get(op->GetPlace()));
}
*/

class BufferSharedInplaceOpPass : public MemoryReusePass {
 protected:
  std::string Type() const override { return "inplace"; }

  void Run(Graph *graph) const override;
};

void BufferSharedInplaceOpPass::Run(Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  bool use_cuda = Get<bool>(kUseCuda);

  // Step 1: Build a reverse map of last_live_ops
  // i.e.: op -> vars
  std::unordered_map<details::ComputationOpHandle *,
                     std::unordered_map<std::string, ir::Node *>>
      candidate_ops;
  for (auto &each_scope_ops : last_live_ops) {
    for (auto &pair : each_scope_ops) {
      // If variable has more than 1 last lived ops, this variable cannot
      // be inplaced.
      if (pair.second.size() != 1) {
        continue;
      }

      auto *op = *(pair.second.begin());
      const std::string &op_type = op->GetOp()->Type();
      const framework::OpDesc *op_desc = op->Node()->Op();
      PADDLE_ENFORCE_NOT_NULL(op_desc);

      auto &infer_inplace = OpInfoMap::Instance().Get(op_type).infer_inplace_;
      if (!infer_inplace) {
        continue;
      }

      const std::string &var_name = pair.first;
      auto in_nodes = this->FindNodesByName(var_name, op->Node()->inputs);
      if (in_nodes.size() == 1) {
        candidate_ops[op][var_name] = *in_nodes.begin();
      }
    }
  }

  // Step 2: Check which vars can be inplaced indeed
  for (auto &op_vars_pair : candidate_ops) {
    auto *op = op_vars_pair.first;
    auto &vars = op_vars_pair.second;

    const std::string &op_type = op->GetOp()->Type();
    auto *op_desc = op->Node()->Op();

    auto in_to_outs =
        OpInfoMap::Instance().Get(op_type).infer_inplace_(*op_desc, use_cuda);
    for (auto &pair : in_to_outs) {
      auto &in_param = pair.first;
      auto &in_args = op_desc->Input(in_param);
      if (in_args.empty()) {
        VLOG(4) << "Cannot inplace because Input(" << in_param
                << ") is empty in " << op_type;
        continue;
      }

      auto &in_arg = in_args[0];
      auto iter = vars.find(in_arg);
      if (iter == vars.end()) {
        VLOG(4) << "Cannot inplace maybe because Input(" << in_param
                << ")=" << in_arg << " is not lastly used in op " << op_type
                << ", or it occurs multiple times in input or occurs in output";
        continue;
      }

      ir::Node *in_node = iter->second;

      auto &out_param = pair.second;
      auto &out_args = op_desc->Output(out_param);

      if (out_args.empty()) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ") is empty in " << op_type;
        continue;
      }

      auto &out_arg = out_args[0];
      auto out_nodes = this->FindNodesByName(out_arg, op->Node()->outputs);
      if (out_nodes.size() != 1) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ")=" << out_arg << " occurs " << out_nodes.size()
                << " time(s) in output of op " << op_type;
        continue;
      }

      auto *out_node = *out_nodes.begin();

      auto &in_var_handle = in_node->Wrapper<details::VarHandleBase>();
      auto &out_var_handle = out_node->Wrapper<details::VarHandleBase>();

      auto *in_var_handle_ptr =
          dynamic_cast<details::VarHandle *>(&in_var_handle);
      auto *out_var_handle_ptr =
          dynamic_cast<details::VarHandle *>(&out_var_handle);

      if (in_var_handle_ptr == nullptr || out_var_handle_ptr == nullptr) {
        continue;
      }

      bool success = this->TryReuseVar(in_var_handle_ptr, out_var_handle_ptr);
      if (success) {
        VLOG(4) << "Inplace performed in op " << op_type << ": "
                << in_var_handle_ptr->Name() << " -> "
                << out_var_handle_ptr->Name()
                << ". Debug String is: " << op->GetOp()->DebugString();
      } else {
        VLOG(4) << "Inplace failed in op " << op_type << ": "
                << in_var_handle_ptr->Name() << " -> "
                << out_var_handle_ptr->Name();
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(buffer_shared_inplace_pass,
              paddle::framework::ir::BufferSharedInplaceOpPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda);
