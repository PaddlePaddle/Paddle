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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/dialect/pd_type_storage.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"

namespace paddle {
namespace dialect {

using VariantType = paddle::variant<bool,
                                    int,
                                    int64_t,
                                    float,
                                    double,
                                    std::string,
                                    std::vector<bool>,
                                    std::vector<int>,
                                    std::vector<int64_t>,
                                    std::vector<float>,
                                    std::vector<double>,
                                    std::vector<std::string>,
                                    phi::Scalar,
                                    std::vector<phi::Scalar>,
                                    phi::IntArray,
                                    phi::DataType,
                                    phi::DataLayout,
                                    phi::Place>;

// TODO(zhangbo): The builtin type needs to cover all data types of
// phi::DataType.
static inline phi::DataType TransToPhiDataType(ir::Type dtype) {
  if (dtype.isa<ir::BFloat16Type>()) {
    return phi::DataType::BFLOAT16;
  } else if (dtype.isa<ir::Float16Type>()) {
    return phi::DataType::FLOAT16;
  } else if (dtype.isa<ir::Float32Type>()) {
    return phi::DataType::FLOAT32;
  } else if (dtype.isa<ir::Float64Type>()) {
    return phi::DataType::FLOAT64;
  } else if (dtype.isa<ir::UInt8Type>()) {
    return phi::DataType::UINT8;
  } else if (dtype.isa<ir::Int8Type>()) {
    return phi::DataType::INT8;
  } else if (dtype.isa<ir::Int16Type>()) {
    return phi::DataType::INT16;
  } else if (dtype.isa<ir::Int32Type>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<ir::Int64Type>()) {
    return phi::DataType::INT64;
  } else if (dtype.isa<ir::BoolType>()) {
    return phi::DataType::BOOL;
  } else if (dtype.isa<ir::Complex64Type>()) {
    return phi::DataType::COMPLEX64;
  } else if (dtype.isa<ir::Complex128Type>()) {
    return phi::DataType::COMPLEX128;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported ir data type when casting it into "
        "phi data type."));
  }
}

static inline ir::Type TransToIrDataType(phi::DataType dtype,
                                         ir::IrContext* ctx = nullptr) {
  if (ctx == nullptr) {
    ctx = ir::IrContext::Instance();
  }
  switch (dtype) {
    case phi::DataType::BFLOAT16:
      return ir::BFloat16Type::get(ctx);
    case phi::DataType::FLOAT16:
      return ir::Float16Type::get(ctx);
    case phi::DataType::FLOAT32:
      return ir::Float32Type::get(ctx);
    case phi::DataType::FLOAT64:
      return ir::Float64Type::get(ctx);
    case phi::DataType::UINT8:
      return ir::UInt8Type::get(ctx);
    case phi::DataType::INT8:
      return ir::Int8Type::get(ctx);
    case phi::DataType::INT16:
      return ir::Int16Type::get(ctx);
    case phi::DataType::INT32:
      return ir::Int32Type::get(ctx);
    case phi::DataType::INT64:
      return ir::Int64Type::get(ctx);
    case phi::DataType::BOOL:
      return ir::BoolType::get(ctx);
    case phi::DataType::COMPLEX64:
      return ir::Complex64Type::get(ctx);
    case phi::DataType::COMPLEX128:
      return ir::Complex128Type::get(ctx);
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported phi data type `%s` when casting it into "
          "ir data type.",
          dtype));
  }
}

static inline ir::Attribute TransToIrAttribute(phi::Scalar scalar,
                                               ir::IrContext* ctx = nullptr) {
  if (ctx == nullptr) {
    ctx = ir::IrContext::Instance();
  }
  switch (scalar.dtype()) {
    case phi::DataType::FLOAT32:
      return ir::FloatAttribute::get(ctx, scalar.to<float>());
    case phi::DataType::FLOAT64:
      return ir::DoubleAttribute::get(ctx, scalar.to<double>());
    case phi::DataType::INT32:
      return ir::Int32Attribute::get(ctx, scalar.to<int32_t>());
    case phi::DataType::INT64:
      return ir::Int64Attribute::get(ctx, scalar.to<int64_t>());
    case phi::DataType::BOOL:
      return ir::BoolAttribute::get(ctx, scalar.to<bool>());
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported phi data type `%s` when casting it into "
          "ir attribute.",
          scalar.dtype()));
  }
}

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
    attr_cast_map = {
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

static inline VariantType GetAttributeData(const ir::Attribute& attr) {
  AttrType attr_type = GetAttributeType(attr);
  return attr_cast_map[attr_type](attr);
}

}  // namespace dialect
}  // namespace paddle
