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

#include "glog/logging.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class BufferSharedInplaceOpPass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "inplace"; }

  void Run(Graph *graph) const override;

  void ApplyImpl(ProgramDesc *main_program,
                 ProgramDesc *startup_program) const override;
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
      if (pair.second.ops().size() != 1) {
        continue;
      }

      auto *op = *(pair.second.ops().begin());
      const std::string &op_type = op->GetOp()->Type();
      const framework::OpDesc *op_desc = op->Node()->Op();
      PADDLE_ENFORCE_NOT_NULL(op_desc,
                              platform::errors::NotFound(
                                  "Op(%s) can not find opdesc.", op->Name()));

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
        OpInfoMap::Instance().Get(op_type).infer_inplace_(use_cuda);
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
                << ". Debug String is: " << op->GetOp()->DebugString()
                << ". ReuseType: " << ReuseType();
      } else {
        VLOG(3) << "Inplace failed in op " << op_type << ": "
                << in_var_handle_ptr->Name() << " -> "
                << out_var_handle_ptr->Name() << ". ReuseType: " << ReuseType();
      }
    }
  }
}

static std::string GetFirstVarName(const OpDesc &op,
                                   const std::string &slot,
                                   bool is_input) {
  const auto &name_map = is_input ? op.Inputs() : op.Outputs();
  auto iter = name_map.find(slot);
  if (iter != name_map.end() && !iter->second.empty()) {
    return iter->second[0];
  }
  return kEmptyVarName;
}

static std::vector<std::vector<std::pair<std::string, std::string>>>
GetInplaceVars(const BlockDesc &block,
               bool use_cuda,
               const std::vector<std::string> &skip_vars,
               const bool &for_partial_block) {
  PADDLE_ENFORCE_EQ(
      block.ID(),
      0,
      platform::errors::Unimplemented("Inplace can only perform in block 0."));
  // only take block 0 gc_vars
  const auto op_gc_vars = GetEagerDeletionCleanVarsForPartial(
      *block.Program(), skip_vars, for_partial_block)[0];
  const auto all_ops = block.AllOps();
  PADDLE_ENFORCE_EQ(op_gc_vars.size(),
                    all_ops.size(),
                    platform::errors::PermissionDenied(
                        "GC analysis error: op number not match."));
  size_t n = all_ops.size();
  std::unordered_set<std::string> visited_vars;
  std::unordered_set<std::string> reused_in_vars(skip_vars.begin(),
                                                 skip_vars.end());
  std::unordered_set<std::string> reused_out_vars(skip_vars.begin(),
                                                  skip_vars.end());
  for (const auto *op : all_ops) {
    if (op->Type() == "share_buffer" || op->Type() == "share_data") {
      const auto &inputs = op->Input("X");
      const auto &outputs = op->Output("Out");
      reused_in_vars.insert(inputs.begin(), inputs.end());
      reused_out_vars.insert(outputs.begin(), outputs.end());
    }
  }

  std::vector<std::vector<std::pair<std::string, std::string>>> result(n);
  for (size_t i = 0; i < n; ++i) {
    const auto &op = *all_ops[i];
    const auto &gc_vars = op_gc_vars[i];
    const auto inputs = op.InputArgumentNames();
    const auto outputs = op.OutputArgumentNames();
    visited_vars.insert(inputs.begin(), inputs.end());

    auto &infer_inplace = OpInfoMap::Instance().Get(op.Type()).infer_inplace_;
    if (gc_vars.empty() || !infer_inplace) {
      visited_vars.insert(outputs.begin(), outputs.end());
      continue;
    }

    const auto var_pair = infer_inplace(use_cuda);
    std::unordered_multiset<std::string> input_set(inputs.begin(),
                                                   inputs.end());
    std::unordered_multiset<std::string> output_set(outputs.begin(),
                                                    outputs.end());
    std::unordered_set<std::string> valid_vars;
    for (const auto &var : gc_vars) {
      if (var != kEmptyVarName && input_set.count(var) == 1 &&
          output_set.count(var) == 0 &&
          block.FindVar(var)->GetType() == proto::VarType::LOD_TENSOR) {
        valid_vars.insert(var);
      }
    }

    if (valid_vars.empty()) {
      visited_vars.insert(outputs.begin(), outputs.end());
      continue;
    }

    for (const auto &pair : var_pair) {
      const auto &input_slot = pair.first;
      const auto &output_slot = pair.second;
      auto input_var = GetFirstVarName(op, input_slot, /*is_input=*/true);
      if (input_var == kEmptyVarName || valid_vars.count(input_var) == 0) {
        continue;
      }
      auto output_var = GetFirstVarName(op, output_slot, /*is_input=*/false);
      if (output_var == kEmptyVarName || visited_vars.count(output_var) > 0) {
        continue;
      }
      auto output_var_desc = block.FindVar(output_var);
      if (output_var_desc == nullptr || output_var_desc->Persistable() ||
          output_var_desc->GetType() != proto::VarType::LOD_TENSOR) {
        continue;
      }

      if (reused_in_vars.count(input_var) > 0 ||
          reused_out_vars.count(output_var) > 0) {
        continue;
      }

      // input_var -> output_var is reusable
      VLOG(10) << "inplace occurs at op " << i << " " << op.Type() << ": "
               << input_var << " -> " << output_var;
      result[i].emplace_back(input_var, output_var);
      reused_in_vars.insert(input_var);
      reused_out_vars.insert(output_var);
    }
    visited_vars.insert(outputs.begin(), outputs.end());
    std::sort(result[i].begin(), result[i].end());
  }
  return result;
}

void BufferSharedInplaceOpPass::ApplyImpl(ProgramDesc *main_program,
                                          ProgramDesc *startup_program) const {
  bool use_cuda = Get<bool>(kUseCuda);
  auto skip_vars = Get<std::vector<std::string>>("mem_opt_skip_vars");
  bool for_partial_block = false;
  if (Has("for_partial_block")) {
    for_partial_block = Get<bool>("for_partial_block");
  }

  auto *block = main_program->MutableBlock(0);
  auto inplace_vars =
      GetInplaceVars(*block, use_cuda, skip_vars, for_partial_block);
  PADDLE_ENFORCE_EQ(inplace_vars.size(),
                    block->OpSize(),
                    platform::errors::PermissionDenied(
                        "Inplace analysis error: op number not match."));
  int64_t n = static_cast<int64_t>(inplace_vars.size());
  for (int64_t i = n - 1; i >= 0; --i) {
    if (inplace_vars[i].empty()) continue;
    auto *op = block->InsertOp(i);
    std::vector<std::string> inputs, outputs;
    inputs.reserve(inplace_vars[i].size());
    outputs.reserve(inplace_vars[i].size());
    for (const auto &pair : inplace_vars[i]) {
      inputs.push_back(pair.first);
      outputs.push_back(pair.second);
    }
    op->SetType("share_buffer");
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
    op->SetOutput("XOut", inputs);  // add necessary dependency
    op->SetAttr("share_dims_and_dtype",
                std::vector<bool>(inputs.size(), false));
  }
  block->Flush();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(buffer_shared_inplace_pass,
              paddle::framework::ir::BufferSharedInplaceOpPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda);
