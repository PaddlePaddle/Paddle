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

#include "paddle/phi/common/complex.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"

namespace pir {
class Type;
class UInt8Type;
class Int8Type;
class Int16Type;
class Int32Type;
class VectorType;
class BFloat16Type;
class Float32Type;
class Float64Type;
class IndexType;
class BoolType;
class Complex64Type;
class Complex128Type;
class StrAttribute;
class BoolAttribute;
class FloatAttribute;
class DoubleAttribute;
class Int32Attribute;
class IndexAttribute;
class Int64Attribute;
class ArrayAttribute;
class PointerAttribute;
class TensorNameAttribute;
class Complex64Attribute;
class Complex128Attribute;

using InsertionPoint = std::pair<Block *, Block::Iterator>;
///
/// \brief Unified interface of the Attribute class. Derivation of all Attribute
/// classes only derives interfaces, not members.
///
class Builder {
 public:
  Builder(IrContext *context,
          Block *block,
          Block::Iterator insertion_point,
          bool forbid_insert_without_position = true)
      : context_(context),
        insertion_point_(block, insertion_point),
        forbid_insert_without_position_(forbid_insert_without_position) {}

  Builder(IrContext *context, bool forbid_insert_without_position)
      : Builder(context,
                nullptr,
                Block::Iterator{},
                forbid_insert_without_position) {}

  Builder(IrContext *context, Block *block)
      : Builder(context, block, block->end()) {}

  explicit Builder(IrContext *context)
      : Builder(context, nullptr, Block::Iterator{}) {}

  Builder(IrContext *context, const InsertionPoint &insert_point)
      : context_(context), insertion_point_(insert_point) {}

  void set_insertion_point(const InsertionPoint &insert_point) {
    insertion_point_ = insert_point;
  }

  /// Set the insert point to the start of the specified block.
  void SetInsertionPointToStart(Block *block) {
    set_insertion_point(block, block->begin());
  }

  /// Set the insertion point to the specified location.
  void set_insertion_point(Block *block, Block::Iterator insert_point) {
    insertion_point_.first = block;
    insertion_point_.second = insert_point;
  }

  /// Set the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void set_insertion_point(Operation *op) {
    set_insertion_point(op->GetParent(), Block::Iterator{*op});
  }

  /// Set the insertion point to the node after the specified operation, which
  /// will cause subsequent insertions to go right after it.
  void SetInsertionPointAfter(Operation *op) {
    set_insertion_point(op->GetParent(), std::next(Block::Iterator{*op}));
  }

  /// Set the insertion point to the end of the specified block.
  void SetInsertionPointToBlockEnd(Block *block) {
    PADDLE_ENFORCE_NOT_NULL(
        block, phi::errors::PreconditionNotMet("argument of block is nullptr"));
    set_insertion_point(block, block->end());
  }

  IrContext *ir_context() const { return context_; }

  Block *block() const { return insertion_point_.first; }

  const InsertionPoint &insertion_point() const { return insertion_point_; }

  /// Creates an operation given the fields represented as an OperationState.
  IR_API Operation *Build(OperationArgument &&argument);

  /// Creates an operation with the given fields.
  IR_API Operation *Build(const std::vector<Value> &inputs,
                          const AttributeMap &attribute,
                          const std::vector<Type> &output_types,
                          pir::OpInfo op_info);

  Operation *Insert(Operation *op);

  /// Create an operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpTy Build(Args &&...args);

  IR_API BoolType bool_type();
  IR_API UInt8Type uint8_type();
  IR_API Int8Type int8_type();
  IR_API Int16Type int16_type();
  IR_API Int32Type int32_type();
  IR_API VectorType vec_type(const std::vector<Type> &);
  IR_API BFloat16Type bfloat16_type();
  IR_API IndexType index_type();
  IR_API Float32Type float32_type();
  IR_API Float64Type float64_type();
  IR_API Complex64Type complex64_type();
  IR_API Complex128Type complex128_type();

  IR_API StrAttribute str_attr(const std::string &value);
  IR_API BoolAttribute bool_attr(bool value);
  IR_API FloatAttribute float_attr(float value);
  IR_API DoubleAttribute double_attr(double value);
  IR_API Int32Attribute int32_attr(int32_t value);
  IR_API IndexAttribute index_attr(int64_t value);
  IR_API Int64Attribute int64_attr(int64_t value);
  IR_API ArrayAttribute array_attr(const std::vector<Attribute> &value);
  IR_API PointerAttribute pointer_attr(void *value);
  IR_API TensorNameAttribute tensor_name_attr(const std::string &value);
  IR_API Complex64Attribute complex64_attr(phi::dtype::complex<float> value);
  IR_API Complex128Attribute complex128_attr(phi::dtype::complex<double> value);

 private:
  IrContext *context_;

  InsertionPoint insertion_point_;

  bool forbid_insert_without_position_;
};

template <typename OpTy, typename... Args>
OpTy Builder::Build(Args &&...args) {
  OperationArgument argument(context_->GetRegisteredOpInfo(OpTy::name()));
  OpTy::Build(*this, argument, std::forward<Args>(args)...);
  Operation *op = Build(std::move(argument));
  return OpTy(op);
}

}  // namespace pir
