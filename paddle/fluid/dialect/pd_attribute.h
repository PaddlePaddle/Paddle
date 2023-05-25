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

#include "paddle/fluid/dialect/pd_attribute_storage.h"
#include "paddle/ir/attribute.h"

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

  phi::IntArray data() const;
};

class ScalarAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(ScalarAttribute, ScalarAttributeStorage);

  bool operator<(const ScalarAttribute &right) const {
    return storage() < right.storage();
  }

  paddle::experimental::Scalar data() const;
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
