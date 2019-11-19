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

namespace paddle {
namespace framework {
namespace ir {

using OpHandleBase = details::OpHandleBase;
using ComputationOpHandle = details::ComputationOpHandle;
using VarHandle = details::VarHandle;
using VarHandleBase = details::VarHandleBase;
using DummyVarHandle = details::DummyVarHandle;

static bool InOutputArgumentOf(const std::string &arg_name,
                               const OperatorBase &op) {
  auto &outputs = op.Outputs();
  for (auto &pair : outputs) {
    auto &out_args = pair.second;
    auto iter = std::find(out_args.begin(), out_args.end(), arg_name);
    if (iter != out_args.end()) return true;
  }
  return false;
}

using IdentityInplaceVarInfo = std::unordered_map<
    VarHandle * /*input_var*/,
    std::vector<std::pair<VarHandle * /*output_var*/,
                          ComputationOpHandle * /*identity op*/>>>;

class BufferSharedIdentityInplaceOpPass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "identity_inplace"; }

  void Run(Graph *graph) const override;

 private:
  IdentityInplaceVarInfo RunOnEachScope(
      const std::unordered_map<std::string, LastLiveOpOfVarInfo> &var_infos)
      const;

  static VarHandle *TryFindMatchedIdentityInplacedOutput(
      const VarHandle &in_var, const ComputationOpHandle &op);
};

IdentityInplaceVarInfo BufferSharedIdentityInplaceOpPass::RunOnEachScope(
    const std::unordered_map<std::string, LastLiveOpOfVarInfo> &var_infos)
    const {
  IdentityInplaceVarInfo identity_inplace_info;
  for (auto &pair : var_infos) {
    auto *in_var = pair.second.var();
    const auto &ops = pair.second.ops();

    std::unordered_set<ComputationOpHandle *> candidate_ops;
    for (auto *op : ops) {
      if (IsIdentityOp(op->GetOp()->Type())) {
        VLOG(5) << "Found identity op " << op->GetOp()->Type();
        candidate_ops.insert(op);
      }
    }

    if (candidate_ops.empty()) continue;
    if (!IsInVarReusable(*in_var)) continue;

    // in_var cannot occur in outputs of any `ops` (not limited to
    // `identity_ops`)!
    bool failure = false;
    auto in_arg = in_var->Name();
    for (auto *op : ops) {
      if (InOutputArgumentOf(in_arg, *(op->GetOp()))) {
        VLOG(5) << "Cannot inplace because " << in_arg
                << " occurs in outputs of some ops";
        failure = true;
        break;
      }
    }

    if (failure) continue;

    for (auto *op : candidate_ops) {
      auto *out_var = TryFindMatchedIdentityInplacedOutput(*in_var, *op);
      if (out_var == nullptr) continue;
      if (!IsInVarReusable(*out_var) || !IsOutVarReusable(*out_var)) {
        VLOG(5) << "Cannot inplace because " << out_var->Name()
                << " is not reusable";
        continue;
      }

      identity_inplace_info[in_var].emplace_back(out_var, op);
    }
  }

  return identity_inplace_info;
}

VarHandle *
BufferSharedIdentityInplaceOpPass::TryFindMatchedIdentityInplacedOutput(
    const VarHandle &in_var, const ComputationOpHandle &op) {
  auto &in_out_pairs = InOutPairOfIdentityOp(op.GetOp()->Type());
  std::string out_slot;
  for (auto &in_to_out : in_out_pairs) {
    auto &in_slot = in_to_out.first;
    auto &inputs = op.GetOp()->Inputs();
    auto in_iter = inputs.find(in_slot);
    if (in_iter == inputs.end()) continue;
    if (in_iter->second.size() != 1) continue;
    if (in_iter->second[0] == in_var.Name()) {
      out_slot = in_to_out.second;
      break;
    }
  }

  if (out_slot.empty()) return nullptr;
  auto &outputs = op.GetOp()->Outputs();
  auto out_iter = outputs.find(out_slot);
  if (out_iter == outputs.end()) return nullptr;
  if (out_iter->second.size() != 1) return nullptr;

  const auto &out_arg = out_iter->second[0];
  auto out_nodes = FindNodesByName(out_arg, op.Node()->outputs);
  if (out_nodes.size() != 1) return nullptr;
  auto *out_node = *(out_nodes.begin());

  PADDLE_ENFORCE_EQ(out_node->IsWrappedBy<VarHandleBase>(), true);
  return dynamic_cast<VarHandle *>(&(out_node->Wrapper<VarHandleBase>()));
}

void BufferSharedIdentityInplaceOpPass::Run(Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  std::vector<IdentityInplaceVarInfo> result;
  result.reserve(last_live_ops.size());
  for (auto &each_scope_var_infos : last_live_ops) {
    result.emplace_back(RunOnEachScope(each_scope_var_infos));
  }

  for (auto &inplace_info : result) {
    for (auto &in_to_out : inplace_info) {
      auto *in_var = in_to_out.first;
      for (auto &op_out_var_pair : in_to_out.second) {
        auto *out_var = op_out_var_pair.first;
        auto *op = op_out_var_pair.second;
        VLOG(5) << "Identity inplace in " << op->GetOp()->Type() << ": "
                << in_var->Name() << " -> " << out_var->Name() << " in scope "
                << in_var->scope_idx();
        AddReuseVar(op, in_var, out_var);
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(buffer_shared_identity_inplace_pass,
              paddle::framework::ir::BufferSharedIdentityInplaceOpPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda);
