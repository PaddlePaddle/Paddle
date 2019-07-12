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

#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include <string>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace operators {

// OpVariant is a wrapper class of OpDesc and OperatorBase
// So that API would be the same.
class OpVariant {
  struct InputsVisitor
      : public boost::static_visitor<const framework::VariableNameMap *> {
    template <typename OpType>
    const framework::VariableNameMap *operator()(const OpType *op) const {
      return &(op->Inputs());
    }
  };

  struct OutputsVisitor
      : public boost::static_visitor<const framework::VariableNameMap *> {
    template <typename OpType>
    const framework::VariableNameMap *operator()(const OpType *op) const {
      return &(op->Outputs());
    }
  };

  struct AttributeMapVisitor
      : public boost::static_visitor<const framework::AttributeMap *> {
    const framework::AttributeMap *operator()(
        const framework::OpDesc *op) const {
      return &(op->GetAttrMap());
    }

    const framework::AttributeMap *operator()(
        const framework::OperatorBase *op) const {
      return &(op->Attrs());
    }
  };

  struct RawPointerVisitor : public boost::static_visitor<const void *> {
    template <typename OpType>
    const void *operator()(const OpType *op) const {
      return op;
    }
  };

 public:
  OpVariant(const framework::OperatorBase *op) : op_(op) {}  // NOLINT

  OpVariant(const framework::OpDesc *op) : op_(op) {}  // NOLINT

  const framework::VariableNameMap &Inputs() const {
    return *boost::apply_visitor(InputsVisitor(), op_);
  }

  const framework::VariableNameMap &Outputs() const {
    return *boost::apply_visitor(OutputsVisitor(), op_);
  }

  const framework::AttributeMap &Attrs() const {
    return *boost::apply_visitor(AttributeMapVisitor(), op_);
  }

  template <typename AttrType>
  const AttrType &Attr(const std::string &name) const {
    auto &attrs = Attrs();
    auto it = attrs.find(name);
    PADDLE_ENFORCE(it != attrs.end(), "Cannot find attribute %s", name);
    return boost::get<AttrType>(it->second);
  }

  bool operator==(const OpVariant &other) const {
    return RawPointer() == other.RawPointer();
  }

  const void *RawPointer() const {
    return boost::apply_visitor(RawPointerVisitor(), op_);
  }

  int which() const { return static_cast<int>(op_.which()); }

  struct Hasher {
    size_t operator()(const OpVariant &op) const {
      return reinterpret_cast<size_t>(op.RawPointer());
    }
  };

 private:
  const boost::variant<const framework::OperatorBase *,
                       const framework::OpDesc *>
      op_;
};

static std::string GetDebugString(const std::vector<std::string> &names) {
  if (names.empty()) return "";
  std::string ret = names[0];
  for (size_t i = 1; i < names.size(); ++i) {
    ret += (" " + names[i]);
  }
  return ret;
}

// Set skip variables of while_op and while_grad_op
// These variables should be skipped when eager deletion enables.
// It is because:
//  1. while_grad_op needs some variables defined in while_op.
//  2. while_grad_op needs variables from the previous time step.
static void SetSkipVars(const OpVariant &op, std::vector<std::string> attr) {
  auto &attrs = const_cast<framework::AttributeMap &>(op.Attrs());
  VLOG(2) << "Prepare to skip " << attr.size()
          << " var(s): " << GetDebugString(attr);
  attrs[kSkipEagerDeletionVars] = std::move(attr);
}

// Check whether the forward while_op and while_grad_op match
// The program may have many while_ops.
static bool IsMatchedWhileOpAndWhileGradOp(const OpVariant &fwd_op,
                                           const OpVariant &grad_op) {
  return fwd_op.Inputs().at(kX) == grad_op.Inputs().at(kX) &&
         fwd_op.Outputs().at(kOutputs) == grad_op.Inputs().at(kOutputs);
}

// Test whether the variable is skippable in forward while_op
// The variable is skippable in while_op when the variable used in while_grad
// is not from grad_block.
static bool IsSkippableVar(const std::string &name,
                           framework::BlockDesc *grad_block) {
  return name != framework::kEmptyVarName && !grad_block->HasVar(name);
}

static void ModifyWhileOpAndWhileGradOpAttr(const OpVariant &fwd_op,
                                            const OpVariant &bwd_op) {
  auto *grad_block = bwd_op.Attr<framework::BlockDesc *>(kStepBlock);

  // Find all skippable variables in forward while_op
  std::unordered_set<std::string> forward_skip_vars;
  for (auto *op_desc : grad_block->AllOps()) {
    for (auto &in_arg_name : op_desc->InputArgumentNames()) {
      if (IsSkippableVar(in_arg_name, grad_block)) {
        forward_skip_vars.insert(in_arg_name);
      }
    }

    for (auto &out_arg_name : op_desc->OutputArgumentNames()) {
      if (IsSkippableVar(out_arg_name, grad_block)) {
        forward_skip_vars.insert(out_arg_name);
      }
    }
  }

  SetSkipVars(fwd_op, std::vector<std::string>(forward_skip_vars.begin(),
                                               forward_skip_vars.end()));

  // Find all skippable variables in while_grad_op
  // The skipped variables are those which would be used across time steps.
  auto &fwd_input = fwd_op.Inputs().at(kX);
  auto &in_grads = bwd_op.Outputs().at(framework::GradVarName(kX));
  PADDLE_ENFORCE_EQ(
      fwd_input.size(), in_grads.size(),
      "Backward input gradient number does not match forward input number.");

  std::unordered_set<std::string> backward_skip_vars;
  for (size_t i = 0; i < in_grads.size(); ++i) {
    if (in_grads[i] == framework::kEmptyVarName) {
      continue;
    }
    backward_skip_vars.insert(in_grads[i]);
    backward_skip_vars.insert(framework::GradVarName(fwd_input[i]));
  }

  SetSkipVars(bwd_op, std::vector<std::string>(backward_skip_vars.begin(),
                                               backward_skip_vars.end()));
}

// Find all while_ops and while_grad_ops in the graph or program
// The while_grad_op and while_op may located in different blocks
// So we should traverse all blocks in the program and find them out.
static void FindAllWhileAndWhileGradOp(std::vector<OpVariant> *while_ops,
                                       std::vector<OpVariant> *while_grad_ops) {
  PADDLE_ENFORCE_GE(while_ops->size(), while_grad_ops->size());

  if (while_ops->empty()) return;

  const auto *program =
      while_ops->front().Attr<framework::BlockDesc *>(kStepBlock)->Program();
  for (size_t i = 1; i < program->Size(); ++i) {
    auto &block = program->Block(i);
    for (size_t j = 0; j < block.OpSize(); ++j) {
      auto *op = block.Op(j);
      if (op->Type() == "while") {
        while_ops->emplace_back(op);
      } else if (op->Type() == "while_grad") {
        while_grad_ops->emplace_back(op);
      }
    }
  }

  PADDLE_ENFORCE_GE(while_ops->size(), while_grad_ops->size(),
                    "There are extra while_grad ops in the graph or program");
}

static void PrepareSafeEagerDeletionOnWhileOpAndWhileGradOpImpl(
    std::vector<OpVariant> *while_ops, std::vector<OpVariant> *while_grad_ops) {
  FindAllWhileAndWhileGradOp(while_ops, while_grad_ops);

  VLOG(2) << "Found while op num: " << while_ops->size()
          << ", while grad op num: " << while_grad_ops->size();

  if (while_grad_ops->empty()) {
    return;
  }

  std::unordered_set<OpVariant, OpVariant::Hasher> while_op_set(
      while_ops->begin(), while_ops->end());

  for (auto &bwd_op : *while_grad_ops) {
    const OpVariant *matched_fwd_op = nullptr;
    for (auto &fwd_op : while_op_set) {
      if (IsMatchedWhileOpAndWhileGradOp(fwd_op, bwd_op)) {
        PADDLE_ENFORCE(matched_fwd_op == nullptr,
                       "Found multiple matched while ops");
        matched_fwd_op = &fwd_op;
      }
    }
    PADDLE_ENFORCE_NOT_NULL(matched_fwd_op,
                            "Cannot find matched forward while op.");
    ModifyWhileOpAndWhileGradOpAttr(*matched_fwd_op, bwd_op);
    while_op_set.erase(*matched_fwd_op);
  }
}

void PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
    int block_id,
    const std::vector<std::unique_ptr<framework::OperatorBase>> &all_ops) {
  // If block_id is not 0, returns
  // This is because all while_ops and while_grad_ops in the whole program
  // would be processed when block_id is 0 (i.e. when Executor::Run() or
  // ParallelExecutor constructs).

  // What's more, all while_ops and while_grad_ops must be processed when
  // block_id is zero. If not, while_op may run first and erase variables
  // used in while_grad_op, and in this moment, while_grad_ops may be not
  // constructed yet.
  if (block_id != 0) return;

  std::vector<OpVariant> fwd_ops, bwd_ops;
  for (auto &op : all_ops) {
    if (op->Type() == "while") {
      fwd_ops.emplace_back(op.get());
    } else if (op->Type() == "while_grad") {
      bwd_ops.emplace_back(op.get());
    }
  }
  PrepareSafeEagerDeletionOnWhileOpAndWhileGradOpImpl(&fwd_ops, &bwd_ops);
}

void PrepareSafeEagerDeletionOnWhileOpAndWhileGradOp(
    const std::vector<framework::OperatorBase *> &while_ops,
    const std::vector<framework::OperatorBase *> &while_grad_ops) {
  std::vector<OpVariant> fwd_ops, bwd_ops;
  fwd_ops.reserve(while_ops.size());
  for (auto *op : while_ops) {
    fwd_ops.emplace_back(op);
  }

  bwd_ops.reserve(while_grad_ops.size());
  for (auto *op : while_grad_ops) {
    bwd_ops.emplace_back(op);
  }

  PrepareSafeEagerDeletionOnWhileOpAndWhileGradOpImpl(&fwd_ops, &bwd_ops);
}

}  // namespace operators
}  // namespace paddle
