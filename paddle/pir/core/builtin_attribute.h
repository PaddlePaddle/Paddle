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

#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/builtin_attribute_storage.h"
#include "paddle/pir/core/utils.h"

namespace pir {
class IR_API BoolAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(BoolAttribute, BoolAttributeStorage);

  bool data() const;
};

class IR_API FloatAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(FloatAttribute, FloatAttributeStorage);

  float data() const;
};

class IR_API DoubleAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DoubleAttribute, DoubleAttributeStorage);

  double data() const;
};

class IR_API Int32Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int32Attribute, Int32AttributeStorage);

  int32_t data() const;
};

class IR_API Int64Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int64Attribute, Int64AttributeStorage);

  int64_t data() const;
};

class IR_API PointerAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(PointerAttribute, PointerAttributeStorage);

  void* data() const;
};

class IR_API TypeAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(TypeAttribute, TypeAttributeStorage);

  Type data() const;
};

class IR_API StrAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(StrAttribute, StrAttributeStorage);

  bool operator<(const StrAttribute& right) const;

  std::string AsString() const;

  size_t size() const;

  static StrAttribute get(IrContext* ctx, const std::string& value);
};

class IR_API ArrayAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(ArrayAttribute, ArrayAttributeStorage);

  std::vector<Attribute> AsVector() const;

  size_t size() const;

  bool empty() const;

  Attribute at(size_t index) const;

  static ArrayAttribute get(IrContext* ctx,
                            const std::vector<Attribute>& value);
};

}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::StrAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::BoolAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::FloatAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::DoubleAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int32Attribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int64Attribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ArrayAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::PointerAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::TypeAttribute)
