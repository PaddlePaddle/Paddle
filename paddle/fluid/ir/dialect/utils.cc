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

#include "paddle/fluid/ir/dialect/utils.h"

namespace paddle {
namespace dialect {

enum class AttrType {
  UNDEFINED = 0,
  BOOL,
  INT32,
  INT64,

  FLOAT,
  DOUBLE,

  ARRAY,
  INT_ARRAY,

  SCALAR,
  DATA_TYPE,
  DATA_LAYOUT,
  PLACE,

  STRING,

  NUM_ATTR_TYPES,
};

static inline AttrType GetAttributeType(const ir::Attribute& attr) {
  if (attr.isa<ir::BoolAttribute>()) {
    return AttrType::BOOL;
  } else if (attr.isa<ir::FloatAttribute>()) {
    return AttrType::FLOAT;
  } else if (attr.isa<ir::DoubleAttribute>()) {
    return AttrType::DOUBLE;
  } else if (attr.isa<ir::Int32Attribute>()) {
    return AttrType::INT32;
  } else if (attr.isa<ir::Int64Attribute>()) {
    return AttrType::INT64;
  } else if (attr.isa<ir::ArrayAttribute>()) {
    return AttrType::ARRAY;
  } else if (attr.isa<ir::StrAttribute>()) {
    return AttrType::STRING;
  } else if (attr.isa<paddle::dialect::IntArrayAttribute>()) {
    return AttrType::INT_ARRAY;
  } else if (attr.isa<paddle::dialect::DataTypeAttribute>()) {
    return AttrType::DATA_TYPE;
  } else if (attr.isa<paddle::dialect::PlaceAttribute>()) {
    return AttrType::PLACE;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported ir Attribute type when casting it into "
        "AttrType."));
  }
}

static std::unordered_map<AttrType,
                          std::function<VariantType(const ir::Attribute& attr)>>
    kAttrCastMap = {
        {AttrType::BOOL,
         [](const ir::Attribute& attr) {
           return VariantType{attr.dyn_cast<ir::BoolAttribute>().data()};
         }},
        {AttrType::FLOAT,
         [](const ir::Attribute& attr) {
           return VariantType{attr.dyn_cast<ir::FloatAttribute>().data()};
         }},
        {AttrType::DOUBLE,
         [](const ir::Attribute& attr) {
           return VariantType{attr.dyn_cast<ir::DoubleAttribute>().data()};
         }},
        {AttrType::INT32,
         [](const ir::Attribute& attr) {
           return VariantType{attr.dyn_cast<ir::Int32Attribute>().data()};
         }},
        {AttrType::INT64,
         [](const ir::Attribute& attr) {
           return VariantType{attr.dyn_cast<ir::Int64Attribute>().data()};
         }},
        {AttrType::INT_ARRAY,
         [](const ir::Attribute& attr) {
           return VariantType{
               attr.dyn_cast<paddle::dialect::IntArrayAttribute>()
                   .data()
                   .GetData()};
         }},
        {AttrType::STRING,
         [](const ir::Attribute& attr) {
           return VariantType{attr.dyn_cast<ir::StrAttribute>().AsString()};
         }},
        {AttrType::DATA_TYPE,
         [](const ir::Attribute& attr) {
           return VariantType{
               attr.dyn_cast<paddle::dialect::DataTypeAttribute>().data()};
         }},
        {AttrType::PLACE,
         [](const ir::Attribute& attr) {
           return VariantType{
               attr.dyn_cast<paddle::dialect::PlaceAttribute>().data()};
         }},
        {AttrType::ARRAY,
         [](const ir::Attribute& attr) {
           auto attr_vec = attr.dyn_cast<ir::ArrayAttribute>().AsVector();
           if (attr_vec.size() == 0) {
             return VariantType{std::vector<int>()};
           }
           AttrType element_type = GetAttributeType(attr_vec[0]);

           if (element_type == AttrType::BOOL) {
             std::vector<bool> vec_bools;
             for (auto vec_element : attr_vec) {
               vec_bools.push_back(
                   vec_element.dyn_cast<ir::BoolAttribute>().data());
             }
             return VariantType{vec_bools};
           } else if (element_type == AttrType::INT32) {
             std::vector<int> vec_int32;
             for (auto vec_element : attr_vec) {
               vec_int32.push_back(
                   vec_element.dyn_cast<ir::Int32Attribute>().data());
             }
             return VariantType{vec_int32};
           } else if (element_type == AttrType::INT64) {
             std::vector<int64_t> vec_int64;
             for (auto vec_element : attr_vec) {
               vec_int64.push_back(
                   vec_element.dyn_cast<ir::Int64Attribute>().data());
             }
             return VariantType{vec_int64};
           } else if (element_type == AttrType::FLOAT) {
             std::vector<float> vec_float;
             for (auto vec_element : attr_vec) {
               vec_float.push_back(
                   vec_element.dyn_cast<ir::FloatAttribute>().data());
             }
             return VariantType{vec_float};
           } else if (element_type == AttrType::DOUBLE) {
             std::vector<double> vec_double;
             for (auto vec_element : attr_vec) {
               vec_double.push_back(
                   vec_element.dyn_cast<ir::DoubleAttribute>().data());
             }
             return VariantType{vec_double};
           } else {
             PADDLE_THROW(phi::errors::Unimplemented(
                 "Unsupported ir Attribute type when casting it into "
                 "vector."));
           }
         }},
};

VariantType GetAttributeData(const ir::Attribute& attr) {
  AttrType attr_type = GetAttributeType(attr);
  return kAttrCastMap[attr_type](attr);
}

}  // namespace dialect
}  // namespace paddle
