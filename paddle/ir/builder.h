// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <list>

#include "paddle/ir/operation.h"

namespace ir {
///
/// \brief Unified interface of the Attribute class. Derivation of all Attribute
/// classes only derives interfaces, not members.
///
class Builder {
 public:
  explicit Builder(IrContext *context) : context_(context) {}
  explicit Builder(Operation *op) : Builder(op->ir_context()) {}

  /// Create an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    OperationArgument argument(context_->GetRegisteredOpInfo(OpTy::name()));
    OpTy::build(*this, argument, std::forward<Args>(args)...);
    Operation *op = Operation::create(argument);
    return dyn_cast<OpTy>(op);
  }

 private:
  IrContext *context_;
  // The current op list this builder is inserting into.
  // After the design of the block data structure is completed,
  // this member will be replaced by the block.
  std::list<Operation *> *op_list_ = nullptr;
  // The insertion point within the list that this builder is inserting before.
  std::list<Operation *>::iterator insertPoint;
};
}  // namespace ir
