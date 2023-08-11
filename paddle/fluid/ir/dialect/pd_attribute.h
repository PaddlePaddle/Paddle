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

#include "paddle/fluid/ir/dialect/pd_attribute_storage.h"
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace dialect {
class IntArrayAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(IntArrayAttribute,
                                    IntArrayAttributeStorage);

  bool operator<(const IntArrayAttribute &right) const {
    return storage() < right.storage();
  }

  const phi::IntArray &data() const;
};

class ScalarAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  static bool classof(ir::Attribute val) {
    return (val.type_id() == ir::BoolAttribute::type_id()) ||
           (val.type_id() == ir::FloatAttribute::type_id()) ||
           (val.type_id() == ir::DoubleAttribute::type_id()) ||
           (val.type_id() == ir::Int32Attribute::type_id()) ||
           (val.type_id() == ir::Int64Attribute::type_id()) ||
           (val.type_id() == ir::StrAttribute::type_id());
  }

  phi::Scalar data();
};

class DataTypeAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataTypeAttribute,
                                    DataTypeAttributeStorage);

  bool operator<(const DataTypeAttribute &right) const {
    return storage() < right.storage();
  }

  phi::DataType data() const;
};

class PlaceAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(PlaceAttribute, PlaceAttributeStorage);

  bool operator<(const PlaceAttribute &right) const {
    return storage() < right.storage();
  }

  phi::Place data() const;
};

class DataLayoutAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataLayoutAttribute,
                                    DataLayoutAttributeStorage);

  bool operator<(const DataLayoutAttribute &right) const {
    return storage() < right.storage();
  }

  phi::DataLayout data() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::IntArrayAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ScalarAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataTypeAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PlaceAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataLayoutAttribute)
