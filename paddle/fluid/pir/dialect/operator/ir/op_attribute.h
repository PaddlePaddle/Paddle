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
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/parser/ir_parser.h"

namespace paddle {
namespace dialect {
class IntArrayAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(IntArrayAttribute,
                                    IntArrayAttributeStorage);

  bool operator<(const IntArrayAttribute &right) const {
    return storage() < right.storage();
  }

  static IntArrayAttribute Parse(pir::IrParser &parser);  // NOLINT

  const phi::IntArray &data() const;
};

class ScalarAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  static bool classof(pir::Attribute val) {
    return (val.type_id() == pir::BoolAttribute::type_id()) ||
           (val.type_id() == pir::FloatAttribute::type_id()) ||
           (val.type_id() == pir::DoubleAttribute::type_id()) ||
           (val.type_id() == pir::Int32Attribute::type_id()) ||
           (val.type_id() == pir::Int64Attribute::type_id()) ||
           (val.type_id() == pir::StrAttribute::type_id());
  }

  static pir::Attribute get(pir::IrContext *ctx, phi::Scalar scalar) {
    return TransToIrAttribute(scalar, ctx);
  }

  phi::Scalar data();
};

class DataTypeAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataTypeAttribute,
                                    DataTypeAttributeStorage);

  bool operator<(const DataTypeAttribute &right) const {
    return storage() < right.storage();
  }

  static DataTypeAttribute Parse(pir::IrParser &parser);  // NOLINT

  phi::DataType data() const;
};

class PlaceAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(PlaceAttribute, PlaceAttributeStorage);

  bool operator<(const PlaceAttribute &right) const {
    return storage() < right.storage();
  }

  static PlaceAttribute Parse(pir::IrParser &parser);  // NOLINT

  phi::Place data() const;
};

class DataLayoutAttribute : public pir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataLayoutAttribute,
                                    DataLayoutAttributeStorage);

  bool operator<(const DataLayoutAttribute &right) const {
    return storage() < right.storage();
  }

  static DataLayoutAttribute Parse(pir::IrParser &parser);  // NOLINT
  phi::DataLayout data() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::IntArrayAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ScalarAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataTypeAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PlaceAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataLayoutAttribute)
