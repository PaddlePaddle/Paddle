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

#include <queue>
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

using OpHandleBase = details::OpHandleBase;
using ComputationOpHandle = details::ComputationOpHandle;
using VarHandle = details::VarHandle;
using VarHandleBase = details::VarHandleBase;

/**
 * The vars in `IdentityInplaceVarInfo` build a `FOREST`.
 *
 * This means that:
 *  1. there is no circle. Otherwise, it must be a bug of Paddle framework.
 *  2. input edge number of any var must be 0 (for root) or 1.
 *  3. output edge number of any var must be 0 (for leaf) or larger than 0.
 *
 * We would use the characteristics of `FOREST` in the following codes,
 * so we prove these 3 conditions here.
 *
 * Condition 1 and 3 is obvious. Let us prove condition 2:
 *
 *  In order to perform in-place reuse, `output_var` must be the 1st version
 *  in SSA graph (it is guaranteed in `MemoryReusePass::IsOutVarReusable`
 *  method). Therefore, if the input edge number of `output_var` is more than 1,
 *  it means `output_var` must be outputs of at least 2 different ops (one op
 *  with the same output is not allowed in Paddle framework). However, if
 *  `output_var` is the output of 2 different ops, `output_var`s in these 2 ops
 *  must have different version, which is guaranteed by SSA graph. Therefore,
 *  `output_var` must be the unique output of unique op. Condition 2 is proved.
 */
using IdentityInplaceVarInfo = std::unordered_map<
    VarHandle * /*input_var*/,
    std::vector<std::pair<VarHandle * /*output_var*/,
                          ComputationOpHandle * /*identity op*/>>>;

/**
 * This method is used to build the adjacent map of the forest inside
 * `info`. The returned result maps a var to its parent var in the forest
 * `info`.
 *
 * For convenience, the root vars of the forest are also one of the key of
 * the output map, but their mapping value are nullptr. This method can
 * also distinguish whether the var is inside the forest or not to avoid
 * probable bug.
 */
static std::unordered_map<VarHandle *, VarHandle *> BuildParent(
    const IdentityInplaceVarInfo &info) {
  std::unordered_map<VarHandle *, VarHandle *> parent;

  for (auto &pair : info) {
    PADDLE_ENFORCE_EQ(pair.second.empty(), false,
                      platform::errors::InvalidArgument(
                          "Var info list cannot be empty, this may be a bug"));

    auto *in_var = pair.first;
    // used to build nullptr parent for each root vars
    parent[in_var];
    for (auto &ovar_op : pair.second) {
      auto *out_var = ovar_op.first;
      parent[out_var] = in_var;
    }
  }
  return parent;
}

/**
 * This method is used to prune the leaf var nodes inside
 * `reused_vars` when branched identity inplace is performed
 * to avoid calculation error.
 */
static void ShrinkIdentityInplacedOps(
    IdentityInplaceVarInfo *info,
    const std::unordered_set<VarHandle *> &reused_vars) {
  if (reused_vars.empty()) {
    return;
  }
  auto parent = BuildParent(*info);

  auto get_parent = [&parent](VarHandle *var) {
    auto iter = parent.find(var);
    PADDLE_ENFORCE_EQ(iter != parent.end(), true,
                      platform::errors::InvalidArgument(
                          "Cannot get parent of variable %s, this may be a bug",
                          var->Name()));
    return iter->second;
  };

  auto get_mutable_info_list = [info](VarHandle *var) {
    auto iter = info->find(var);
    PADDLE_ENFORCE_EQ(
        iter != info->end(), true,
        platform::errors::InvalidArgument(
            "Cannot get info list of variable %s, this may be a bug",
            var->Name()));
    return &(iter->second);
  };

  for (auto *var : reused_vars) {
    bool is_branch_reused = false;

    auto *cur_var = var;
    auto *parent_var = get_parent(cur_var);
    while (parent_var != nullptr) {
      auto *info_list = get_mutable_info_list(parent_var);
      if (info_list->size() > 1) {
        is_branch_reused = true;
        break;
      }
      cur_var = parent_var;
      parent_var = get_parent(cur_var);
    }

    if (!is_branch_reused) {
      continue;
    }

    VLOG(5) << "Remove reused leaf var " << var->Name() << " on scope "
            << var->scope_idx();
    parent_var = get_parent(var);
    parent.erase(var);
    auto *target_list = get_mutable_info_list(parent_var);
    target_list->erase(
        std::remove_if(
            target_list->begin(), target_list->end(),
            [var](const std::pair<VarHandle *, ComputationOpHandle *> &p) {
              return p.first == var;
            }),
        target_list->end());
    if (target_list->empty()) {
      info->erase(parent_var);
    }
  }
}

static std::unordered_set<VarHandle *>
GetUnReusableLeafVarsAfterIdentityInplaced(
    const IdentityInplaceVarInfo &info,
    const std::unordered_set<VarHandle *> &is_read_by_other_ops) {
  auto parent = BuildParent(info);

  std::unordered_set<VarHandle *> unreusable_vars;
  for (auto &pair : parent) {
    if (pair.second) {  // non-root
      continue;
    }

    std::unordered_set<VarHandle *> leaves;
    bool var_is_read_by_other_ops = false;
    bool var_is_branched_reuse = false;
    auto *cur_var = pair.first;
    std::queue<VarHandle *> q;
    q.push(cur_var);

    while (!q.empty()) {
      cur_var = q.front();
      q.pop();

      auto iter = info.find(cur_var);
      if (iter == info.end()) {  // leaf var
        leaves.insert(cur_var);
      } else {
        if (is_read_by_other_ops.count(cur_var) > 0) {
          var_is_read_by_other_ops = true;
        }

        if (iter->second.size() > 1) {
          var_is_branched_reuse = true;
        }

        for (auto &ovar_op : iter->second) {
          q.push(ovar_op.first);
        }
      }
    }

    if (var_is_read_by_other_ops || var_is_branched_reuse) {
      unreusable_vars.insert(leaves.begin(), leaves.end());
    }
  }
  return unreusable_vars;
}

class BufferSharedIdentityInplaceOpPass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "identity_inplace"; }

  void Run(Graph *graph) const override;

 private:
  std::pair<IdentityInplaceVarInfo, std::unordered_set<VarHandle *>>
  RunOnEachScope(const std::unordered_map<std::string, LastLiveOpOfVarInfo>
                     &var_infos) const;

  static VarHandle *TryFindMatchedIdentityInplacedOutput(
      const VarHandle &in_var, const ComputationOpHandle &op);
};

std::pair<IdentityInplaceVarInfo, std::unordered_set<VarHandle *>>
BufferSharedIdentityInplaceOpPass::RunOnEachScope(
    const std::unordered_map<std::string, LastLiveOpOfVarInfo> &var_infos)
    const {
  IdentityInplaceVarInfo identity_inplace_info;
  std::unordered_set<VarHandle *> is_read_by_other_ops;
  std::unordered_set<VarHandle *> collected_out_vars;

  /**
   * The memory of some out var has been reused by non-identity ops,
   * this vars must be forced to be leaf when identity inplaced is performed.
   * If branched identity inplace is performed (many vars may reuse the memory
   * of the same var), this kind of leaf nodes must be pruned (otherwise,
   * caculation result may be wrong).
   * If non-branched identity inplace is performed, this kind of leaf nodes
   * should not be pruned.
   */
  std::unordered_set<VarHandle *> reused_out_vars;
  for (auto &pair : var_infos) {
    auto *in_var = pair.second.var();
    if (!IsInVarReusable(*in_var)) continue;

    // force leaf vars
    if (reused_out_vars.count(in_var) > 0) continue;

    auto in_arg = in_var->Name();

    auto last_live_ops = pair.second.ops();
    /**
     * If all `ops` only read `in_var` and do not write `in_var`,
     * we should scan all output nodes of `in_var`. If any output
     * node of `in_var` is an identity op node, in-place can be
     * performed.
     */
    bool failure = false;
    for (auto *op : last_live_ops) {
      failure = FindNodesByName(in_arg, op->Node()->inputs).empty() ||
                !FindNodesByName(in_arg, op->Node()->outputs).empty();
      if (failure) break;
    }

    if (failure) continue;

    std::unordered_set<ComputationOpHandle *> candidate_ops;
    for (auto *op_node : in_var->Node()->outputs) {
      auto *op = dynamic_cast<ComputationOpHandle *>(
          &(op_node->Wrapper<OpHandleBase>()));
      if (op == nullptr) {
        candidate_ops.clear();
        break;
      }
      if (IsIdentityOp(op->GetOp()->Type())) {
        VLOG(5) << "Found identity op " << op->GetOp()->Type();
        candidate_ops.insert(op);
      }
    }

    if (candidate_ops.empty()) continue;

    for (auto *op : candidate_ops) {
      auto *out_var = TryFindMatchedIdentityInplacedOutput(*in_var, *op);
      if (out_var == nullptr) continue;
      if (!IsOutVarReusable(*out_var)) {
        VLOG(5) << "Cannot inplace because " << out_var->Name()
                << " is not reusable";
        continue;
      }

      if (!IsInVarReusable(*out_var)) {
        reused_out_vars.insert(out_var);
      }

      // `identity_inplace_info` is a forest, so that `out_var` should
      // has 1 parent `in_var` atmost.
      PADDLE_ENFORCE_EQ(collected_out_vars.count(out_var), 0,
                        platform::errors::InvalidArgument(
                            "Out var %s occurs twice when building "
                            "identity_inplace_info, this may be a bug",
                            out_var->Name()));
      identity_inplace_info[in_var].emplace_back(out_var, op);
      collected_out_vars.insert(out_var);
      last_live_ops.erase(op);
    }

    // If last live ops of `in_var` is not a subset of identity in-placed
    // ops, the leaf vars of such ops must be non-reusable. Otherwise,
    // the calculation results may be wrong after CrossOpMemoryReusePass.
    // Those vars must be marked as "non-reusable"!
    if (!last_live_ops.empty()) {
      is_read_by_other_ops.insert(in_var);
    }
  }

  ShrinkIdentityInplacedOps(&identity_inplace_info, reused_out_vars);
  auto non_reusable_vars = GetUnReusableLeafVarsAfterIdentityInplaced(
      identity_inplace_info, is_read_by_other_ops);
  return std::make_pair(std::move(identity_inplace_info),
                        std::move(non_reusable_vars));
}

VarHandle *
BufferSharedIdentityInplaceOpPass::TryFindMatchedIdentityInplacedOutput(
    const VarHandle &in_var, const ComputationOpHandle &op) {
  auto &in_out_pairs = InOutPairOfIdentityOp(op.GetOp()->Type());
  std::unique_ptr<std::string> out_slot;
  for (auto &in_to_out : in_out_pairs) {
    auto &in_slot = in_to_out.first;
    auto &inputs = op.GetOp()->Inputs();
    auto in_iter = inputs.find(in_slot);
    if (in_iter == inputs.end()) continue;
    if (in_iter->second.size() != 1) continue;
    if (in_iter->second[0] == in_var.Name()) {
      out_slot.reset(new std::string(in_to_out.second));
      break;
    }
  }

  if (!out_slot) return nullptr;
  auto &outputs = op.GetOp()->Outputs();
  auto out_iter = outputs.find(*out_slot);
  if (out_iter == outputs.end()) return nullptr;
  if (out_iter->second.size() != 1) return nullptr;

  const auto &out_arg = out_iter->second[0];
  auto out_nodes = FindNodesByName(out_arg, op.Node()->outputs);
  if (out_nodes.size() != 1) return nullptr;
  auto *out_node = *(out_nodes.begin());

  PADDLE_ENFORCE_EQ(out_node->IsWrappedBy<VarHandleBase>(), true,
                    platform::errors::InvalidArgument(
                        "out node must be wrapped by VarHandleBase"));
  return dynamic_cast<VarHandle *>(&(out_node->Wrapper<VarHandleBase>()));
}

void BufferSharedIdentityInplaceOpPass::Run(Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  std::vector<IdentityInplaceVarInfo> inplace_infos;
  inplace_infos.reserve(last_live_ops.size());

  std::vector<std::unordered_set<VarHandle *>> non_reusable_leaf_vars;
  non_reusable_leaf_vars.reserve(last_live_ops.size());

  for (auto &each_scope_var_infos : last_live_ops) {
    auto each_scope_ret = RunOnEachScope(each_scope_var_infos);
    inplace_infos.emplace_back(std::move(each_scope_ret.first));
    non_reusable_leaf_vars.emplace_back(std::move(each_scope_ret.second));
  }

  for (auto &inplace_info : inplace_infos) {
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

  for (auto &vars_each_scope : non_reusable_leaf_vars) {
    for (auto *each_var : vars_each_scope) {
      VLOG(1) << "Mark " << each_var->Name() << " on scope "
              << each_var->scope_idx() << " as not reusable";
      MarkAsNotReusableInVar(*each_var);
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
    .RequirePassAttr(paddle::framework::ir::kUseCuda)
    .RequirePassAttr(paddle::framework::ir::kSkipReuseVars);
