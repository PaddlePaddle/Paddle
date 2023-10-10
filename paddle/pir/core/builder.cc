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

#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/region.h"
#include "paddle/pir/core/value.h"

namespace pir {
/// Create an operation given the fields represented as an OperationState.
Operation *Builder::Build(OperationArgument &&argument) {
  return Insert(Operation::Create(std::move(argument)));
}

/// Creates an operation with the given fields.
Operation *Builder::Build(const std::vector<Value> &inputs,
                          const AttributeMap &attribute,
                          const std::vector<Type> &output_types,
                          OpInfo op_info) {
  return Build(OperationArgument(inputs, attribute, output_types, op_info));
}

Operation *Builder::Insert(Operation *op) {
  if (insert_point_.first) {
    insert_point_.first->insert(insert_point_.second, op);
  } else {
    LOG(WARNING) << "Builder's Block is nullptr, insert failed.";
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
Int64Attribute Builder::int64_attr(int64_t value) {
  return Int64Attribute::get(context_, value);
}
ArrayAttribute Builder::array_attr(const std::vector<Attribute> &value) {
  return ArrayAttribute::get(context_, value);
}
PointerAttribute Builder::pointer_attr(void *value) {
  return PointerAttribute::get(context_, value);
}

}  // namespace pir
