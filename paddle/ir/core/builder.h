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
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"

namespace ir {
class Type;
class UInt8Type;
class Int8Type;
class VectorType;
class BFloat16Type;
class Float32Type;
class Float64Type;
class Int16Type;
class IndexType;
class BoolType;
class Complex64Type;
class Complex128Type;
class StrAttribute;
class BoolAttribute;
class FloatAttribute;
class DoubleAttribute;
class Int32Attribute;
class Int64Attribute;
class ArrayAttribute;
class PointerAttribute;

///
/// \brief Unified interface of the Attribute class. Derivation of all Attribute
/// classes only derives interfaces, not members.
///
class Builder {
 public:
  Builder(IrContext *context, Block *block, Block::iterator insert_point)
      : context_(context) {
    SetInsertionPoint(block, insert_point);
  }

  Builder(IrContext *context, Block *block)
      : Builder(context, block, block->end()) {}

  explicit Builder(IrContext *context)
      : Builder(context, nullptr, Block::iterator{}) {}

  /// Set the insertion point to the specified location.
  void SetInsertionPoint(Block *block, Block::iterator insert_point) {
    // TODO(liuyuanle): check that insertPoint is in this rather than some other
    // block.
    this->block_ = block;
    this->insert_point_ = insert_point;
  }

  /// Set the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void SetInsertionPoint(Operation *op) {
    SetInsertionPoint(op->GetParent(), Block::iterator{*op});
  }

  /// Set the insertion point to the node after the specified operation, which
  /// will cause subsequent insertions to go right after it.
  void SetInsertionPointAfter(Operation *op) {
    SetInsertionPoint(op->GetParent(), std::next(Block::iterator{*op}));
  }

  /// Set the insertion point to the start of the specified block.
  void SetInsertionPointToStart(Block *block) {
    SetInsertionPoint(block, block->begin());
  }

  /// Set the insertion point to the end of the specified block.
  void SetInsertionPointToEnd(Block *block) {
    SetInsertionPoint(block, block->end());
  }

  IrContext *ir_context() const { return context_; }

  Block *block() const { return block_; }

  /// Creates an operation given the fields represented as an OperationState.
  IR_API Operation *Build(OperationArgument &&argument);

  /// Creates an operation with the given fields.
  IR_API Operation *Build(const std::vector<ir::OpResult> &inputs,
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

  IR_API UInt8Type uint8_type();
  IR_API Int8Type int8_type();
  IR_API VectorType vec_type(const std::vector<Type> &);
  IR_API BFloat16Type bfloat16_type();
  IR_API IndexType index_type();
  IR_API Float32Type float32_type();
  IR_API Float64Type float64_type();
  IR_API Int16Type int16_type();
  IR_API BoolType bool_type();
  IR_API Complex64Type complex64_type();
  IR_API Complex128Type complex128_type();

  IR_API StrAttribute str_attr(const std::string &value);
  IR_API BoolAttribute bool_attr(bool value);
  IR_API FloatAttribute float_attr(float value);
  IR_API DoubleAttribute double_attr(double value);
  IR_API Int32Attribute int32_attr(int32_t value);
  IR_API Int64Attribute int64_attr(int64_t value);
  IR_API ArrayAttribute array_attr(const std::vector<Attribute> &value);
  IR_API PointerAttribute pointer_attr(void *value);

 private:
  Operation *Insert(Operation *op);

  IrContext *context_;
  Block *block_;
  // The insertion point within the list that this builder is inserting before.
  Block::iterator insert_point_;
};

}  // namespace ir
