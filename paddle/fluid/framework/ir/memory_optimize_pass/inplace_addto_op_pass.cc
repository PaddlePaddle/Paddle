// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

class InplaceAddToOpPass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "inplace_addto"; }

  void Run(Graph *graph) const override;

 private:
  // 1. Add last living op of in_var, add any last living op of out_var
  // 2. Set reference count of in_var to be 2
  void UpdateLastLiveOpOfVar(details::ComputationOpHandle *op,
                             details::VarHandle *in_var,
                             details::VarHandle *out_var) const override {
    size_t scope_idx = op->GetScopeIdx();
    auto *last_live_ops_of_vars_ =
        &Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);
    auto *var_infos_ = &(Get<MemOptVarInfoMapList>(kMemOptVarInfoMapList));
    auto out_var_op_iter =
        (*last_live_ops_of_vars_)[scope_idx].find(out_var->Name());

    // In Reduce mode, some output variable(gradient of parameter) does not have
    // last live ops
    details::ComputationOpHandle *last_live_op_of_in_var = nullptr;
    if (out_var_op_iter == (*last_live_ops_of_vars_)[scope_idx].end()) {
      last_live_op_of_in_var = op;
    } else {
      PADDLE_ENFORCE_EQ(
          out_var_op_iter->second.ops().empty(), false,
          platform::errors::InvalidArgument(
              "Var(%s)'s last live op should not empty.", out_var->Name()));
      last_live_op_of_in_var = *(out_var_op_iter->second.ops().begin());
    }

    auto *last_live_ops_of_in_var =
        (*last_live_ops_of_vars_)[scope_idx][in_var->Name()].mutable_ops();
    // last_live_ops_of_in_var->clear();
    last_live_ops_of_in_var->insert(last_live_op_of_in_var);

    auto in_var_info_iter = (*var_infos_)[scope_idx].find(in_var->Name());
    PADDLE_ENFORCE_NE(
        in_var_info_iter, (*var_infos_)[scope_idx].end(),
        platform::errors::NotFound("Cannot find variable %s.", in_var->Name()));

    in_var_info_iter->second->SetRefCnt(2);  // before inplace, it is 1
  }
};

void InplaceAddToOpPass::Run(Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  bool use_cuda = Get<bool>(kUseCuda);

  // Currently, only perform InplaceAddToOpPass on cuda place
  if (!use_cuda) {
    return;
  }

  // Step 1: Build a reverse map of last_live_ops
  // i.e.: op -> vars
  std::unordered_map<details::ComputationOpHandle *,
                     std::unordered_map<std::string, ir::Node *>>
      candidate_ops;
  for (auto &each_scope_ops : last_live_ops) {
    for (auto &pair : each_scope_ops) {
      // If variable has more than 1 last lived ops, this variable cannot
      // be inplaced.
      if (pair.second.ops().size() != 1) {
        continue;
      }

      auto *op = *(pair.second.ops().begin());
      const std::string &op_type = op->GetOp()->Type();
      const framework::OpDesc *op_desc = op->Node()->Op();
      PADDLE_ENFORCE_NOT_NULL(
          op_desc, platform::errors::NotFound("Op(%s) can not find opdesc.",
                                              op->Name()));

      // only grad op should be processed.
      if (op_type != "grad_add") {
        continue;
      }

      const std::string &var_name = pair.first;
      auto in_nodes = this->FindNodesByName(var_name, op->Node()->inputs);
      if (in_nodes.size() == 1) {
        candidate_ops[op][var_name] = *in_nodes.begin();
      }
      VLOG(4) << "Find op " << op_type << " with input(" << var_name
              << ") that can do inplace add to";
    }
  }

  // Step 2: Check which vars can be inplaced indeed
  for (auto &op_vars_pair : candidate_ops) {
    auto *op = op_vars_pair.first;

    // The original gradient accumulation is g = sum(g_0, g_1,..., g_n), and it
    // could be changed as follws if inplace addto is enabled:
    // g_sum_0 = g_0
    // g_sum_1 = grad_add(g_sum_0, g_1)
    // g_sum_2 = grad_add(g_sum_1, g_2)
    // ...
    // g_sum_n = grad_add(g_sum_n-1, g_n)

    // here we will add inplace for each grad_add, for example, for the first
    // grad_add, g_sum_0 -> g1, g_sum_1 -> g1, and set grad_add as skipped.

    const std::string &op_type = op->GetOp()->Type();

    PADDLE_ENFORCE_EQ(op->Node()->inputs.size(), 2,
                      platform::errors::InvalidArgument(
                          "The size of inputs of %s should be 2, but got %d",
                          op_type, op->Node()->inputs.size()));

    PADDLE_ENFORCE_EQ(op->Node()->outputs.size(), 1,
                      platform::errors::InvalidArgument(
                          "The size of outputs of %s should be 1, but got %d",
                          op_type, op->Node()->outputs.size()));

    auto *left_var_ptr = dynamic_cast<details::VarHandle *>(
        &(op->Node()->inputs[0]->Wrapper<details::VarHandleBase>()));
    auto *right_var_ptr = dynamic_cast<details::VarHandle *>(
        &(op->Node()->inputs[1]->Wrapper<details::VarHandleBase>()));
    auto *out_var_ptr = dynamic_cast<details::VarHandle *>(
        &(op->Node()->outputs[0]->Wrapper<details::VarHandleBase>()));

    if (left_var_ptr == nullptr || right_var_ptr == nullptr ||
        out_var_ptr == nullptr) {
      continue;
    }

    // auto *left_generated_op = dynamic_cast<details::ComputationOpHandle *>(
    //     left_var_ptr->GeneratedOp());

    auto *right_generated_op = dynamic_cast<details::ComputationOpHandle *>(
        right_var_ptr->GeneratedOp());

    auto *out_generated_op = dynamic_cast<details::ComputationOpHandle *>(
        out_var_ptr->GeneratedOp());

    // NOTE(zhiqiu): currently, only conv2d_grad supports addto strategy
    if (right_generated_op->Name() != "conv2d_grad") {
      continue;
    }

    // NOTE(zhiqiu): Normally, if we inplace a->b, we should let a generated
    // before b. However, in the situation of inplace addto, we do not care
    // the order, since a+b is equal to b+a. Is there any exception for that?

    // AddDependencyVar(right_generated_op, left_generated_op);
    // no need, as discussed above.

    // step (a): inplace right_var->left_var of grad_add

    this->AddReuseVar(right_generated_op, left_var_ptr, right_var_ptr);
    UpdateLastLiveOpOfVar(right_generated_op, left_var_ptr, right_var_ptr);
    VLOG(4) << "Inplace performed in op " << right_generated_op->GetOp()->Type()
            << ": " << left_var_ptr->Name() << " -> " << right_var_ptr->Name()
            << ". Debug String is: "
            << right_generated_op->GetOp()->DebugString()
            << ". ReuseType: " << ReuseType();

    // step (b): inplace out -> right_var of grad_add

    this->AddReuseVar(out_generated_op, right_var_ptr, out_var_ptr, true);

    VLOG(4) << "Inplace performed in op " << op_type << ": "
            << left_var_ptr->Name() << " -> " << out_var_ptr->Name()
            << ". Debug String is: " << op->GetOp()->DebugString()
            << ". ReuseType: " << ReuseType();

    // step (c): make right_var cannot inplace afterwards. canbe done
    // aotomatically since CollectReusedVars is called before any reuse.

    // step (d): make right_var's generated op use addto
    right_generated_op->GetOp()->SetAttr("use_addto", true);

    // step (e): make grad_add skip running
    op->SetSkipRunning(true);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_addto_op_pass, paddle::framework::ir::InplaceAddToOpPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda);
