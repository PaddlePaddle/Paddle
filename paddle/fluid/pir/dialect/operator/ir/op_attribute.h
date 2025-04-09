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

#include "paddle/fluid/pir/dialect/operator/ir/attribute_storage.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/dll_decl.h"

namespace paddle {
namespace dialect {
// __force_backend__ in ["gpu","gpudnn","cpu",""]
inline const char kForceBackendAttr[] = "__force_backend__";
inline const char kCanRunTrtAttr[] = "__l_trt__";

class IR_API IntArrayAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(IntArrayAttribute,
                                    IntArrayAttributeStorage);

  bool operator<(const IntArrayAttribute &right) const {
    return storage() < right.storage();
  }

  const phi::IntArray &data() const;

  static std::string name() { return "a_intarray"; }
};

class ScalarAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  static bool classof(pir::Attribute val) {
    return (val.type_id() == pir::BoolAttribute::type_id()) ||
           (val.type_id() == pir::FloatAttribute::type_id()) ||
           (val.type_id() == pir::DoubleAttribute::type_id()) ||
           (val.type_id() == pir::Int32Attribute::type_id()) ||
           (val.type_id() == pir::IndexAttribute::type_id()) ||
           (val.type_id() == pir::Int64Attribute::type_id()) ||
           (val.type_id() == pir::StrAttribute::type_id()) ||
           (val.type_id() == pir::Complex64Attribute::type_id()) ||
           (val.type_id() == pir::Complex128Attribute::type_id());
  }

  static pir::Attribute get(pir::IrContext *ctx, phi::Scalar scalar) {
    return TransToIrAttribute(scalar, ctx);
  }

  phi::Scalar data() const;

  static std::string name() { return "a_scalar"; }
};

class IR_API DataTypeAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataTypeAttribute,
                                    DataTypeAttributeStorage);

  bool operator<(const DataTypeAttribute &right) const {
    return storage() < right.storage();
  }

  phi::DataType data() const;

  static std::string name() { return "a_dtype"; }
};

class PlaceAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(PlaceAttribute, PlaceAttributeStorage);

  bool operator<(const PlaceAttribute &right) const {
    return storage() < right.storage();
  }

  phi::Place data() const;
  static std::string name() { return "a_place"; }
};

class DataLayoutAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataLayoutAttribute,
                                    DataLayoutAttributeStorage);

  bool operator<(const DataLayoutAttribute &right) const {
    return storage() < right.storage();
  }

  phi::DataLayout data() const;
  static std::string name() { return "a_layout"; }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::IntArrayAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ScalarAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataTypeAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PlaceAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataLayoutAttribute)
