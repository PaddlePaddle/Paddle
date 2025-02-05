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

#include "paddle/phi/common/complex.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/builtin_attribute_storage.h"
#include "paddle/pir/include/core/utils.h"

namespace pir {
class IR_API BoolAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(BoolAttribute, BoolAttributeStorage);

  static std::string name() { return "a_bool"; }
  bool data() const;
};

class IR_API Complex64Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Complex64Attribute,
                                    Complex64AttributeStorage);

  static std::string name() { return "a_c64"; }
  phi::dtype::complex<float> data() const;
};

class IR_API Complex128Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Complex128Attribute,
                                    Complex128AttributeStorage);

  static std::string name() { return "a_c128"; }
  phi::dtype::complex<double> data() const;
};

class IR_API FloatAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(FloatAttribute, FloatAttributeStorage);

  static std::string name() { return "a_f32"; }
  float data() const;
};

class IR_API DoubleAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DoubleAttribute, DoubleAttributeStorage);

  static std::string name() { return "a_f64"; }
  double data() const;
};

class IR_API Int32Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int32Attribute, Int32AttributeStorage);

  static std::string name() { return "a_i32"; }
  int32_t data() const;
};

class IR_API IndexAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(IndexAttribute, IndexAttributeStorage);

  static std::string name() { return "a_index"; }
  int64_t data() const;
};

class IR_API Int64Attribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(Int64Attribute, Int64AttributeStorage);

  static std::string name() { return "a_i64"; }
  int64_t data() const;
};

class IR_API PointerAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(PointerAttribute, PointerAttributeStorage);

  static std::string name() { return "a_pointer"; }
  void* data() const;
};

class IR_API TypeAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(TypeAttribute, TypeAttributeStorage);

  static std::string name() { return "a_type"; }
  Type data() const;
};

class IR_API StrAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(StrAttribute, StrAttributeStorage);

  bool operator<(const StrAttribute& right) const;

  std::string AsString() const;

  static std::string name() { return "a_str"; }
  size_t size() const;

  static StrAttribute get(IrContext* ctx, const std::string& value);
};

class IR_API ArrayAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(ArrayAttribute, ArrayAttributeStorage);

  std::vector<Attribute> AsVector() const;
  static std::string name() { return "a_array"; }

  size_t size() const;

  bool empty() const;

  // Returns element at specified location pos, with bounds checking.
  Attribute at(size_t index) const;

  // Returns element at specified location pos. No bounds checking is performed.
  Attribute operator[](size_t index) const;

  static ArrayAttribute get(IrContext* ctx,
                            const std::vector<Attribute>& value);
};

class IR_API TensorNameAttribute : public Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(TensorNameAttribute, StrAttributeStorage);

  bool operator<(const TensorNameAttribute& right) const;
  static std::string name() { return "a_tensorname"; }
  std::string data() const;

  size_t size() const;

  static TensorNameAttribute get(IrContext* ctx,
                                 const std::string& tensor_name);
};

}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::StrAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::BoolAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::FloatAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::DoubleAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int32Attribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Int64Attribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::IndexAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ArrayAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::PointerAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::TypeAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::TensorNameAttribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Complex64Attribute)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::Complex128Attribute)
