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

#include "paddle/fluid/operators/controlflow/pylayer_op_helper.h"

#include <string>

namespace paddle::framework {
class ProgramDesc;
}  // namespace paddle::framework

namespace paddle::operators {

static bool IsMatchedPyLayerOpAndPyLayerGradOp(const OpVariant &fwd_op,
                                               const OpVariant &bwd_op) {
  return fwd_op.Outputs().at(PyLayerOp::kScope) ==
         bwd_op.Inputs().at(PyLayerOp::kScope);
}

static void FindAllPyLayerOpAndPyLayerGradOp(
    const framework::ProgramDesc &program,
    std::vector<OpVariant> *fwd_ops,
    std::vector<OpVariant> *bwd_ops) {
  PADDLE_ENFORCE_GE(
      fwd_ops->size(),
      bwd_ops->size(),
      common::errors::InvalidArgument(
          "Size of forward ops must be greater or equal to backward ops. The "
          "number of forward ops is %d and the number of backward ops is %d",
          fwd_ops->size(),
          bwd_ops->size()));

  for (size_t i = 1; i < program.Size(); ++i) {
    auto &block = program.Block(i);
    for (size_t j = 0; j < block.OpSize(); ++j) {
      auto *op = block.Op(static_cast<int>(j));
      if (op->Type() == "pylayer") {
        fwd_ops->emplace_back(op);
      } else if (op->Type() == "pylayer_grad") {
        bwd_ops->emplace_back(op);
      }
    }
  }

  PADDLE_ENFORCE_GE(
      fwd_ops->size(),
      bwd_ops->size(),
      common::errors::InvalidArgument(
          "There are more pylayer_grad ops than "
          "pylayer ops in the graph or program. The number of "
          "forward ops is %d and the number of backward ops is %d",
          fwd_ops->size(),
          bwd_ops->size()));
}

static void SetSkipVarsForPyLayerOp(OpVariant *fwd_op, OpVariant *bwd_op) {
  auto *grad_block = bwd_op->Attr<framework::BlockDesc *>("backward_block");
  auto is_skippable_in_fwd = [grad_block](const std::string &var_name) {
    return var_name != framework::kEmptyVarName &&
           !grad_block->HasVar(var_name);
  };

  std::unordered_set<std::string> forward_skip_vars;
  for (auto *op_desc : grad_block->AllOps()) {
    for (auto &in_arg_name : op_desc->InputArgumentNames()) {
      if (is_skippable_in_fwd(in_arg_name)) {
        forward_skip_vars.insert(in_arg_name);
      }
    }

    for (auto &out_arg_name : op_desc->OutputArgumentNames()) {
      if (is_skippable_in_fwd(out_arg_name)) {
        forward_skip_vars.insert(out_arg_name);
      }
    }
  }

  auto &fwd_attrs = const_cast<framework::AttributeMap &>(fwd_op->Attrs());
  std::vector<std::string> skip_vars_vec(forward_skip_vars.begin(),
                                         forward_skip_vars.end());
  VLOG(2) << "Prepare to skip " << skip_vars_vec.size()
          << " var(s): " << string::join_strings(skip_vars_vec, ' ');
  fwd_attrs[PyLayerOp::kSkipEagerDeletionVars] = std::move(skip_vars_vec);
}

static void PrepareSafeEagerDeletionOnPyLayerOpAndPyLayerGradOp(
    const framework::ProgramDesc &program,
    std::vector<OpVariant> *pylayer_ops,
    std::vector<OpVariant> *pylayer_grad_ops) {
  FindAllPyLayerOpAndPyLayerGradOp(program, pylayer_ops, pylayer_grad_ops);

  VLOG(2) << "Found pylayer op num: " << pylayer_ops->size()
          << ", pylayer_grad op num: " << pylayer_grad_ops->size();

  if (pylayer_grad_ops->empty()) {
    return;
  }

  std::unordered_set<OpVariant, OpVariant::Hasher> pylayer_op_set(
      pylayer_ops->begin(), pylayer_ops->end());

  for (auto &bwd_op : *pylayer_grad_ops) {
    const OpVariant *matched_fwd_op = nullptr;
    for (auto &fwd_op : pylayer_op_set) {
      if (IsMatchedPyLayerOpAndPyLayerGradOp(fwd_op, bwd_op)) {
        PADDLE_ENFORCE_EQ(matched_fwd_op,
                          nullptr,
                          common::errors::PreconditionNotMet(
                              "Found multiple matched pylayer ops."));
        matched_fwd_op = &fwd_op;
      }
    }

    PADDLE_ENFORCE_NOT_NULL(matched_fwd_op,
                            common::errors::PreconditionNotMet(
                                "Cannot find matched forward pylayer op."));

    SetSkipVarsForPyLayerOp(const_cast<OpVariant *>(matched_fwd_op), &bwd_op);
    pylayer_op_set.erase(*matched_fwd_op);
  }
}

void PrepareSafeEagerDeletionOnPyLayerOpAndPyLayerGradOp(
    const framework::ProgramDesc &program,
    int block_id,
    const std::vector<std::unique_ptr<framework::OperatorBase>> &all_ops) {
  // If block_id is not 0, returns
  // This is because all pylayer_ops and pylayer_grad_ops
  // in the whole program would be processed when block_id is 0 (i.e.
  // when Executor::Run() or ParallelExecutor constructs).

  // What's more, all pylayer_ops and pylayer_grad_ops
  // must be processed when block_id is zero. If not, pylayer_op
  // may run first and erase variables used in pylayer_grad_op,
  // and in this moment, pylayer_grad_ops may be not constructed yet.
  if (block_id != 0) return;

  std::vector<OpVariant> fwd_ops, bwd_ops;
  for (auto &op : all_ops) {
    if (op->Type() == "pylayer") {
      fwd_ops.emplace_back(op.get());
    } else if (op->Type() == "pylayer_grad") {
      bwd_ops.emplace_back(op.get());
    }
  }

  PrepareSafeEagerDeletionOnPyLayerOpAndPyLayerGradOp(
      program, &fwd_ops, &bwd_ops);
}

void PrepareSafeEagerDeletionOnPyLayerOpAndPyLayerGradOp(
    const framework::ProgramDesc &program,
    const std::vector<OpVariant> &pylayer_ops,
    const std::vector<OpVariant> &pylayer_grad_ops) {
  std::vector<OpVariant> fwd_ops = pylayer_ops;
  std::vector<OpVariant> bwd_ops = pylayer_grad_ops;

  PrepareSafeEagerDeletionOnPyLayerOpAndPyLayerGradOp(
      program, &fwd_ops, &bwd_ops);
}

}  // namespace paddle::operators
