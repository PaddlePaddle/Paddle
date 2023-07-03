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

#include "paddle/ir/core/type.h"
#include "paddle/ir/core/type_base.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_type_storage.h"

namespace ir {
namespace pdl {

class PDLType : public ir::Type {
 public:
  using Type::Type;
  DECLARE_TYPE_UTILITY_FUNCTOR(PDLType, ir::TypeStorage);
};

class TypeType : public PDLType {
 public:
  using PDLType::PDLType;
  DECLARE_TYPE_UTILITY_FUNCTOR(TypeType, ir::TypeStorage);
};

class ValueType : public PDLType {
 public:
  using PDLType::PDLType;
  DECLARE_TYPE_UTILITY_FUNCTOR(ValueType, ir::TypeStorage);
};

class AttributeType : public PDLType {
 public:
  using PDLType::PDLType;
  DECLARE_TYPE_UTILITY_FUNCTOR(AttributeType, ir::TypeStorage);
};

class OperationType : public PDLType {
 public:
  using PDLType::PDLType;
  DECLARE_TYPE_UTILITY_FUNCTOR(OperationType, ir::TypeStorage);
};

class RangeType : public PDLType {
 public:
  using PDLType::PDLType;
  DECLARE_TYPE_UTILITY_FUNCTOR(RangeType, detail::RangeTypeStorage);

  Type getElementType() const { return storage()->element_type_; }
};

}  // namespace pdl
}  // namespace ir

IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDLType);
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::ValueType);
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::TypeType);
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::AttributeType);
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::OperationType);
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::RangeType);
