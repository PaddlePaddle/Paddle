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
#include "paddle/fluid/operators/recurrent_op.h"

namespace paddle {
namespace operators {

static bool IsMatchedRecurrentOpAndRecurrentGradOp(const OpVariant &fwd_op,
                                                   const OpVariant &grad_op) {
  return fwd_op.Inputs().at(RecurrentBase::kInputs) ==
             grad_op.Inputs().at(RecurrentBase::kInputs) &&
         fwd_op.Outputs().at(RecurrentBase::kOutputs) ==
             grad_op.Inputs().at(RecurrentBase::kOutputs);
}

// Returns whether the variable is skippable in forward recurrent op
// The variable is skippable in recurrent_op when the variable used in
// recurrent_grad is not from grad_block.
static bool IsSkippableVar(const std::string &name,
                           framework::BlockDesc *grad_block) {
  return name != framework::kEmptyVarName && !grad_block->HasVar(name);
}

static void ClearSkipVars(const OpVariant &op) {
  auto &attrs = const_cast<framework::AttributeMap &>(op.Attrs());
  std::vector<std::string> &attr_skip_vars = BOOST_GET(
      std::vector<std::string>, attrs[RecurrentBase::kSkipEagerDeletionVars]);
  attr_skip_vars.clear();
}

// Add skip vars into op's attribute
template <class Container>
static void AddSkipVars(const OpVariant &op, const Container &skip_vars) {
  auto &attrs = const_cast<framework::AttributeMap &>(op.Attrs());
  VLOG(2) << "Prepare to add " << skip_vars.size()
          << " skip var(s): " << paddle::string::join_strings(skip_vars, ' ');
  std::vector<std::string> &attr_skip_vars = BOOST_GET(
      std::vector<std::string>, attrs[RecurrentBase::kSkipEagerDeletionVars]);
  attr_skip_vars.insert(attr_skip_vars.end(), skip_vars.cbegin(),
                        skip_vars.cend());
}

// Find all ops and grad ops with given type name. The ops and grad ops
// may locate in different blocks so we should traverse all blocks in the
// program and find them out
static void FindAllOpAndGradOp(const framework::ProgramDesc &program,
                               OpAndGradOpPair *op_and_grad_op,
                               const std::string &type_name,
                               const std::string &backward_type_name) {
  OpVariantSet &ops = op_and_grad_op->first;
  OpVariantSet &grad_ops = op_and_grad_op->second;

  PADDLE_ENFORCE_GE(
      ops.size(), grad_ops.size(),
      platform::errors::InvalidArgument(
          "There are more grad ops than forward ops in the graph or program, "
          "the number of ops is %d and the number of grad_ops is %d.",
          ops.size(), grad_ops.size()));

  for (size_t i = 1; i < program.Size(); ++i) {
    auto &block = program.Block(i);
    for (size_t j = 0; j < block.OpSize(); ++j) {
      auto *op = block.Op(j);
      if (op->Type() == type_name) {
        ops.emplace(op);
      } else if (op->Type() == backward_type_name) {
        grad_ops.emplace(op);
      }
    }
  }

  PADDLE_ENFORCE_GE(
      ops.size(), grad_ops.size(),
      platform::errors::InvalidArgument(
          "There are more grad ops than forward ops in the graph or program, "
          "the number of ops is %d and the number of grad_ops is %d.",
          ops.size(), grad_ops.size()));
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

// Add memory vars in recurrent op as skip vars.
static void AddOpMemVarsAsSkip(const OpVariant &op, bool set_grad_mem_vars) {
  bool has_state = op.Attr<bool>(RecurrentBase::kHasStates);
  if (has_state) {
    std::unordered_set<std::string> skip_vars;

    auto &mem_vars = op.Attr<std::vector<std::string>>(RecurrentBase::kStates);
    skip_vars.insert(mem_vars.begin(), mem_vars.end());

    auto &pre_mem_vars =
        op.Attr<std::vector<std::string>>(RecurrentBase::kExStates);
    skip_vars.insert(pre_mem_vars.begin(), pre_mem_vars.end());

    if (set_grad_mem_vars) {
      auto mem_grad_vars = GradVarLists(mem_vars);
      skip_vars.insert(mem_grad_vars.begin(), mem_grad_vars.end());
      auto pre_mem_grad_vars = GradVarLists(pre_mem_vars);
      skip_vars.insert(pre_mem_grad_vars.begin(), pre_mem_grad_vars.end());
    }
    AddSkipVars(op, skip_vars);
  }
}

// Set outputs and memory vars of the input forward op as skip vars
static void SetRecurrentForwardOpOnlySkipVarAttr(const OpVariant &fwd_op) {
  ClearSkipVars(fwd_op);

  AddOpMemVarsAsSkip(fwd_op, /* set_grad_mem_vars = */ false);
  auto &output_vars = fwd_op.Outputs().at(RecurrentBase::kOutputs);
  AddSkipVars(fwd_op, output_vars);
}

// Set skip vars of matched recurrent op and recurrent_grad op
static void SetRecurrentOpAndRecurrentGradOpSkipVarAttr(
    const OpVariant &fwd_op, const OpVariant &bwd_op) {
  // Find all skippable variables in forward recurrent_op
  ClearSkipVars(fwd_op);
  AddOpMemVarsAsSkip(fwd_op, /* set_grad_mem_vars = */ false);

  auto *grad_block =
      bwd_op.Attr<framework::BlockDesc *>(RecurrentBase::kStepBlock);
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
  AddSkipVars(fwd_op, fwd_skip_vars);

  // Find all skippable variables in recurrent_grad_op
  // The skippable variables are those which would be used across time steps
  ClearSkipVars(bwd_op);
  AddOpMemVarsAsSkip(bwd_op, /* set_grad_mem_vars = */ true);
  std::unordered_set<std::string> bwd_skip_vars;

  auto &fwd_input = fwd_op.Inputs().at(RecurrentBase::kInputs);
  auto &in_grads =
      bwd_op.Outputs().at(framework::GradVarName(RecurrentBase::kInputs));

  PADDLE_ENFORCE_EQ(
      fwd_input.size(), in_grads.size(),
      platform::errors::PreconditionNotMet(
          "Backward input gradient number does not match forward "
          "input number. The number of forward input number is %d and the "
          "number of backward input gradient number is %d.",
          fwd_input.size(), in_grads.size()));
  for (size_t i = 0; i < in_grads.size(); ++i) {
    if (in_grads[i] == framework::kEmptyVarName) {
      continue;
    }
    bwd_skip_vars.insert(in_grads[i]);
    bwd_skip_vars.insert(framework::GradVarName(fwd_input[i]));
  }

  auto &fwd_param = fwd_op.Inputs().at(RecurrentBase::kParameters);
  auto &param_grads =
      bwd_op.Outputs().at(framework::GradVarName(RecurrentBase::kParameters));
  PADDLE_ENFORCE_EQ(
      fwd_param.size(), param_grads.size(),
      platform::errors::PreconditionNotMet(
          "Backward parameter gradient number does not match "
          "forward parameter number. The number of forward parameter number is "
          "%d and the number of backward parameter gradient is %d.",
          fwd_param.size(), param_grads.size()));
  for (size_t i = 0; i < fwd_param.size(); ++i) {
    if (param_grads[i] == framework::kEmptyVarName) {
      continue;
    }
    bwd_skip_vars.insert(param_grads[i]);
    bwd_skip_vars.insert(framework::GradVarName(fwd_param[i]));
  }

  AddSkipVars(bwd_op, bwd_skip_vars);
}

void PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
    const framework::ProgramDesc &program, int block_id,
    const std::vector<std::unique_ptr<paddle::framework::OperatorBase>>
        &all_ops) {
  // If block_id is not 0, returns
  // This is because all recurrent_ops and recurrent_grad_ops in the whole
  // program would be processed when block_id is 0 (i.e. when Executor::Run()
  // or ParallelExecutor constructs).

  // What's more, all recurrent_ops and recurrent_grad_ops must be processed
  // when block_id is zero. If not, recurrent_op may run first and erase
  // variables
  // used in recurrent_grad_op, and in this moment, recurrent_grad_ops may be
  // not constructed yet.
  if (block_id != 0) return;

  OpAndGradOpPair op_pair;
  for (auto &op : all_ops) {
    if (op->Type() == "recurrent") {
      op_pair.first.emplace(op.get());
    } else if (op->Type() == "recurrent_grad") {
      op_pair.second.emplace(op.get());
    }
  }
  PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(program, &op_pair);
}

void PrepareSafeEagerDeletionOnRecurrentOpAndRecurrentGradOp(
    const framework::ProgramDesc &program, OpAndGradOpPair *op_pair) {
  // Find all ops and grad ops at all blocks
  FindAllOpAndGradOp(program, op_pair, "recurrent", "recurrent_grad");

  OpVariantSet &recurrent_ops = op_pair->first;
  OpVariantSet &recurrent_grad_ops = op_pair->second;

  VLOG(2) << "Found recurrent op num: " << recurrent_ops.size()
          << ", recurrent grad op num: " << recurrent_grad_ops.size();

  if (recurrent_ops.empty()) {
    return;
  }

  for (auto &bwd_op : recurrent_grad_ops) {
    const OpVariant *matched_fwd_op = nullptr;
    for (auto &fwd_op : recurrent_ops) {
      if (IsMatchedRecurrentOpAndRecurrentGradOp(fwd_op, bwd_op)) {
        PADDLE_ENFORCE_EQ(matched_fwd_op, nullptr,
                          platform::errors::PreconditionNotMet(
                              "Found multiple recurrent forward op matches "
                              "recurrent grad op."));
        matched_fwd_op = &fwd_op;
      }
    }
    PADDLE_ENFORCE_NOT_NULL(matched_fwd_op,
                            platform::errors::PreconditionNotMet(
                                "Cannot find matched forward op."));
    SetRecurrentOpAndRecurrentGradOpSkipVarAttr(*matched_fwd_op, bwd_op);
    recurrent_ops.erase(*matched_fwd_op);
  }

  for (auto &fwd_op : recurrent_ops) {
    SetRecurrentForwardOpOnlySkipVarAttr(fwd_op);
  }
}

}  // namespace operators
}  // namespace paddle
