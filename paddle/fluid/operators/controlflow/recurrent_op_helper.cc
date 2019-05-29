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

#include "paddle/fluid/operators/controlflow/recurrent_op_helper.h"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace operators {

static constexpr char kStepBlock[] = "sub_block";
static constexpr char kStepScopes[] = "step_scopes";
static constexpr char kInputs[] = "inputs";
static constexpr char kInputsGrad[] = "inputs@GRAD";
static constexpr char kOutputs[] = "outputs";
static constexpr char kOutputsGrad[] = "outputs@GRAD";
static constexpr char kParameters[] = "parameters";
static constexpr char kHasStates[] = "has_states";
static constexpr char kExStates[] = "ex_states";
static constexpr char kStates[] = "states";
static constexpr char kSkipEagerDeletionVars[] = "skip_eager_deletion_vars";

static bool IsMatchedRecurrentOpAndRecurrentGradOp(const OpVariant &fwd_op,
                                                   const OpVariant &grad_op) {
  return fwd_op.Inputs().at(kInputs) == grad_op.Inputs().at(kInputs) &&
         fwd_op.Outputs().at(kOutputs) == grad_op.Inputs().at(kOutputs);
}

// Returns whether the variable is skippable in forward recurrent op
// The variable is skippable in recurrent_op when the variable used in
// recurrent_grad
// is not from grad_block.
static bool IsSkippableVar(const std::string &name,
                           framework::BlockDesc *grad_block) {
  return name != framework::kEmptyVarName && !grad_block->HasVar(name);
}

// Add skip vars into op's attribute
static void SetSkipVars(const OpVariant &op,
                        const std::unordered_set<std::string> &skip_vars) {
  auto &attrs = const_cast<framework::AttributeMap &>(op.Attrs());
  VLOG(2) << "Prepare to skip " << skip_vars.size()
          << " var(s): " << paddle::string::join_strings(skip_vars, ' ');
  // attrs[kSkipEagerDeletionVars] =
  // std::vector<std::string>(skip_vars.cbegin(), skip_vars.cend());
  std::vector<std::string> &attr_skip_vars =
      boost::get<std::vector<std::string>>(attrs[kSkipEagerDeletionVars]);
  attr_skip_vars.insert(attr_skip_vars.end(), skip_vars.cbegin(),
                        skip_vars.cend());
}

// Find all ops and grad ops with given type name. The ops and grad ops
// may locate in different blocks so we should traverse all blocks in the
// program and find them out
static void FindAllOpAndGradOp(OpAndGradOpPair *op_and_grad_op,
                               const std::string &type_name) {
  OpVariantSet &ops = op_and_grad_op->first;
  OpVariantSet &grad_ops = op_and_grad_op->second;

  PADDLE_ENFORCE_GE(ops.size(), grad_ops.size(),
                    "There are extra grad ops in the graph or program");

  if (ops.empty()) return;

  const auto *program =
      ops.begin()->Attr<framework::BlockDesc *>(kStepBlock)->Program();
  for (size_t i = 1; i < program->Size(); ++i) {
    auto &block = program->Block(i);
    for (size_t j = 0; j < block.OpSize(); ++j) {
      auto *op = block.Op(j);
      if (op->Type() == type_name) {
        ops.emplace(op);
      } else if (op->Type() == (type_name + "_grad")) {
        grad_ops.emplace(op);
      }
    }
  }

  PADDLE_ENFORCE_GE(ops.size(), grad_ops.size(),
                    "There are extra grad ops in the graph or program");
}

// Returns GradVarName of input var names
static std::vector<std::string> GradVarLists(
    const std::vector<std::string> &var_names) {
  std::vector<std::string> retv;
  retv.reserve(var_names.size());
  std::transform(var_names.begin(), var_names.end(), std::back_inserter(retv),
                 framework::GradVarName);
  return retv;
}

// Set memory vars in recurrent op as skip vars.
static void SetOpMemVarsAsSkip(const OpVariant &op, bool set_grad) {
  bool has_state = op.Attr<bool>(kHasStates);
  if (has_state) {
    std::unordered_set<std::string> skip_vars;

    auto &mem_vars = op.Attr<std::vector<std::string>>(kStates);
    skip_vars.insert(mem_vars.begin(), mem_vars.end());

    auto &pre_mem_vars = op.Attr<std::vector<std::string>>(kExStates);
    skip_vars.insert(pre_mem_vars.begin(), pre_mem_vars.end());

    if (set_grad) {
      auto mem_grad_vars = GradVarLists(mem_vars);
      skip_vars.insert(mem_grad_vars.begin(), mem_grad_vars.end());
      auto pre_mem_grad_vars = GradVarLists(pre_mem_vars);
      skip_vars.insert(pre_mem_grad_vars.begin(), pre_mem_grad_vars.end());
    }
    SetSkipVars(op, skip_vars);
  }
}

// Set outputs and memory vars of the input forward op as skip vars
static void SetRecurrentForwardOpOnlySkipVarAttr(const OpVariant &fwd_op) {
  SetOpMemVarsAsSkip(fwd_op, false);

  std::unordered_set<std::string> fwd_skip_vars;
  auto &output_vars = fwd_op.Outputs().at(kOutputs);
  for (const std::string &name : output_vars) {
    fwd_skip_vars.insert(name);
  }
  SetSkipVars(fwd_op, fwd_skip_vars);
}

// Set skip vars of matched recurrent op and recurrent_grad op
static void SetRecurrentOpAndRecurrentGradOpSkipVarAttr(
    const OpVariant &fwd_op, const OpVariant &bwd_op) {
  // Find all skippable variables in forward recurrent_op
  SetOpMemVarsAsSkip(fwd_op, false);

  auto *grad_block = bwd_op.Attr<framework::BlockDesc *>(kStepBlock);
  std::unordered_set<std::string> fwd_skip_vars;
  for (auto *op_desc : grad_block->AllOps()) {
    for (auto &in_arg_name : op_desc->InputArgumentNames()) {
      if (IsSkippableVar(in_arg_name, grad_block)) {
        fwd_skip_vars.insert(in_arg_name);
      }
    }
    for (auto &out_arg_name : op_desc->OutputArgumentNames()) {
      if (IsSkippableVar(out_arg_name, grad_block)) {
        fwd_skip_vars.insert(out_arg_name);
      }
    }
  }
  SetSkipVars(fwd_op, fwd_skip_vars);

  // Find all skippable variables in recurrent_grad_op
  // The skippable variables are those which would be used across time steps
  SetOpMemVarsAsSkip(bwd_op, true);
  std::unordered_set<std::string> bwd_skip_vars;

  auto &fwd_input = fwd_op.Inputs().at(kInputs);
  auto &in_grads = bwd_op.Outputs().at(framework::GradVarName(kInputs));
  // auto &param_grads =
  // bwd_op.Outputs().at(framework::GradVarName(kParameters));

  PADDLE_ENFORCE_EQ(
      fwd_input.size(), in_grads.size(),
      "Backward input gradient number does not match forward input number.");
  for (size_t i = 0; i < in_grads.size(); ++i) {
    if (in_grads[i] == framework::kEmptyVarName) {
      continue;
    }
    bwd_skip_vars.insert(in_grads[i]);
    bwd_skip_vars.insert(framework::GradVarName(fwd_input[i]));
  }

  auto &fwd_param = fwd_op.Inputs().at(kParameters);
  auto &param_grads = bwd_op.Outputs().at(framework::GradVarName(kParameters));
  PADDLE_ENFORCE_EQ(fwd_param.size(), param_grads.size(),
                    "Backward parameter gradient number does not match forward "
                    "parameter number.");
  for (size_t i = 0; i < fwd_param.size(); ++i) {
    if (param_grads[i] == framework::kEmptyVarName) {
      continue;
    }
    bwd_skip_vars.insert(param_grads[i]);
    bwd_skip_vars.insert(framework::GradVarName(fwd_param[i]));
  }

  SetSkipVars(bwd_op, bwd_skip_vars);
}

void PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
    int block_id, const std::vector<std::unique_ptr<OperatorBase>> &all_ops) {
  // If block_id is not 0, returns
  // This is because all while_ops and while_grad_ops in the whole program
  // would be processed when block_id is 0 (i.e. when Executor::Run() or
  // ParallelExecutor constructs).

  // What's more, all while_ops and while_grad_ops must be processed when
  // block_id is zero. If not, while_op may run first and erase variables
  // used in while_grad_op, and in this moment, while_grad_ops may be not
  // constructed yet.
  if (block_id != 0) return;

  OpAndGradOpPair op_pair;
  for (auto &op : all_ops) {
    if (op->Type() == "recurrent") {
      op_pair.first.emplace(op.get());
    } else if (op->Type() == "recurrent_grad") {
      op_pair.second.emplace(op.get());
    }
  }
  PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(&op_pair);
}

void PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
    OpAndGradOpPair *op_pair) {
  // Find all ops and grad ops at all blocks
  FindAllOpAndGradOp(op_pair, "recurrent");

  OpVariantSet &recurrent_ops = op_pair->first;
  OpVariantSet &recurrent_grad_ops = op_pair->second;

  VLOG(2) << "Found recurrent op num: " << recurrent_ops.size()
          << ", recurrent grad op num: " << recurrent_grad_ops.size();

  if (recurrent_ops.empty()) {
    return;
  }

  if (recurrent_grad_ops.empty()) {
    for (auto &fwd_op : recurrent_ops) {
      SetRecurrentForwardOpOnlySkipVarAttr(fwd_op);
    }
    return;
  }

  for (auto &bwd_op : recurrent_grad_ops) {
    const OpVariant *matched_fwd_op = nullptr;
    for (auto &fwd_op : recurrent_ops) {
      if (IsMatchedRecurrentOpAndRecurrentGradOp(fwd_op, bwd_op)) {
        PADDLE_ENFORCE(matched_fwd_op == nullptr,
                       "Found multiple matched recurrent op");
        matched_fwd_op = &fwd_op;
      }
    }
    PADDLE_ENFORCE_NOT_NULL(matched_fwd_op, "Cannot find matched forward op");
    SetRecurrentOpAndRecurrentGradOpSkipVarAttr(*matched_fwd_op, bwd_op);
    recurrent_ops.erase(*matched_fwd_op);
  }
}

}  // namespace operators
}  // namespace paddle
