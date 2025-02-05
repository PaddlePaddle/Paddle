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

#include <glog/logging.h>

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/value.h"

namespace pir {
/// Create an operation given the fields represented as an OperationState.
Operation *Builder::Build(OperationArgument &&argument) {
  Operation *op = Insert(Operation::Create(std::move(argument)));
  // TODO(ljz): Generalize here to be a hook function in the future.
  // we add op_role attribute only when it is not equal to -1.
  if (op_role_ != -1) {
    op->set_attribute("op_role", Int32Attribute::get(context_, op_role_));
  }
  if (chunk_id_ != -1) {
    op->set_attribute("chunk_id", Int32Attribute::get(context_, chunk_id_));
  }
  return op;
}

/// Creates an operation with the given fields.
Operation *Builder::Build(const std::vector<Value> &inputs,
                          const AttributeMap &attribute,
                          const std::vector<Type> &output_types,
                          OpInfo op_info) {
  return Build(OperationArgument(inputs, attribute, output_types, op_info));
}

Operation *Builder::Insert(Operation *op) {
  if (insertion_point_.first) {
    insertion_point_.first->insert(insertion_point_.second, op);
  } else if (forbid_insert_without_position_) {
    IR_THROW("Insertion position not set, insert failed.");
  }
  return op;
}

BoolType Builder::bool_type() { return BoolType::get(context_); }
UInt8Type Builder::uint8_type() { return UInt8Type::get(context_); }
Int8Type Builder::int8_type() { return Int8Type::get(context_); }
Int16Type Builder::int16_type() { return Int16Type::get(context_); }
Int32Type Builder::int32_type() { return Int32Type::get(context_); }
VectorType Builder::vec_type(const std::vector<Type> &value) {
  return VectorType::get(context_, value);
}
BFloat16Type Builder::bfloat16_type() { return BFloat16Type::get(context_); }
Float32Type Builder::float32_type() { return Float32Type::get(context_); }

Float64Type Builder::float64_type() { return Float64Type::get(context_); }
IndexType Builder::index_type() { return IndexType::get(context_); }
Complex64Type Builder::complex64_type() { return Complex64Type::get(context_); }
Complex128Type Builder::complex128_type() {
  return Complex128Type::get(context_);
}
Float8E4M3FNType Builder::float8e4m3fn_type() {
  return Float8E4M3FNType::get(context_);
}
Float8E5M2Type Builder::float8e5m2_type() {
  return Float8E5M2Type::get(context_);
}
StrAttribute Builder::str_attr(const std::string &value) {
  return StrAttribute::get(context_, value);
}
BoolAttribute Builder::bool_attr(bool value) {
  return BoolAttribute::get(context_, value);
}
FloatAttribute Builder::float_attr(float value) {
  return FloatAttribute::get(context_, value);
}
DoubleAttribute Builder::double_attr(double value) {
  return DoubleAttribute::get(context_, value);
}
Int32Attribute Builder::int32_attr(int32_t value) {
  return Int32Attribute::get(context_, value);
}
IndexAttribute Builder::index_attr(int64_t value) {
  return IndexAttribute::get(context_, value);
}
Int64Attribute Builder::int64_attr(int64_t value) {
  return Int64Attribute::get(context_, value);
}
ArrayAttribute Builder::array_attr(const std::vector<Attribute> &value) {
  return ArrayAttribute::get(context_, value);
}
PointerAttribute Builder::pointer_attr(void *value) {
  return PointerAttribute::get(context_, value);
}
TensorNameAttribute Builder::tensor_name_attr(const std::string &value) {
  return TensorNameAttribute::get(context_, value);
}

BuilderAttrGuard::BuilderAttrGuard(std::shared_ptr<Builder> builder,
                                   int op_role,
                                   int chunk_id)
    : builder_(builder),
      pre_op_role_(builder_->op_role()),
      pre_chunk_id_(builder_->chunk_id()) {
  if (pre_op_role_ != op_role) {
    builder_->set_op_role(op_role);
  }
  if (pre_chunk_id_ != chunk_id) {
    builder_->set_chunk_id(chunk_id);
  }
}

BuilderAttrGuard::~BuilderAttrGuard() {  // NOLINT
  builder_->set_op_role(pre_op_role_);
  builder_->set_chunk_id(pre_chunk_id_);
}

}  // namespace pir
