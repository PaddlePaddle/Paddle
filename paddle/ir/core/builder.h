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

#include "paddle/ir/core/block.h"
#include "paddle/ir/core/operation.h"

namespace ir {
///
/// \brief Unified interface of the Attribute class. Derivation of all Attribute
/// classes only derives interfaces, not members.
///
class Builder {
 public:
  Builder(IrContext *context, Block *block, Block::iterator insert_point)
      : context_(context), block_(block), insert_point_(insert_point) {}
  Builder(IrContext *context, Block *block)
      : Builder(context, block, block->end()) {}

  IrContext *context() const { return context_; }

  Block *block() const { return block_; }

  /// Creates an operation given the fields represented as an OperationState.
  Operation *Build(OperationArgument &&argument);

  /// Creates an operation with the given fields.
  Operation *Build(const std::vector<ir::OpResult> &inputs,
                   const AttributeMap &attribute,
                   const std::vector<ir::Type> &output_types,
                   ir::OpInfo op_info);

  /// Create an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy Build(Args &&...args) {
    OperationArgument argument(context_->GetRegisteredOpInfo(OpTy::name()));
    OpTy::Build(*this, argument, std::forward<Args>(args)...);
    Operation *op = Build(std::move(argument));
    return op->dyn_cast<OpTy>();
  }

 private:
  Operation *Insert(Operation *op);

  IrContext *context_;
  Block *block_ = nullptr;
  // The insertion point within the list that this builder is inserting before.
  Block::iterator insert_point_;
};
}  // namespace ir
