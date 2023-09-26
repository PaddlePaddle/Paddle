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

// #include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/type_storage.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_type.h"

namespace paddle {
namespace dialect {

using VariantType = phi::Attribute;

// TODO(zhangbo): The builtin type needs to cover all data types of
// phi::DataType.
static inline phi::DataType TransToPhiDataType(pir::Type dtype) {
  if (dtype.isa<pir::BFloat16Type>()) {
    return phi::DataType::BFLOAT16;
  } else if (dtype.isa<pir::Float16Type>()) {
    return phi::DataType::FLOAT16;
  } else if (dtype.isa<pir::Float32Type>()) {
    return phi::DataType::FLOAT32;
  } else if (dtype.isa<pir::Float64Type>()) {
    return phi::DataType::FLOAT64;
  } else if (dtype.isa<pir::UInt8Type>()) {
    return phi::DataType::UINT8;
  } else if (dtype.isa<pir::Int8Type>()) {
    return phi::DataType::INT8;
  } else if (dtype.isa<pir::Int16Type>()) {
    return phi::DataType::INT16;
  } else if (dtype.isa<pir::Int32Type>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<pir::Int64Type>()) {
    return phi::DataType::INT64;
  } else if (dtype.isa<pir::IndexType>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<pir::BoolType>()) {
    return phi::DataType::BOOL;
  } else if (dtype.isa<pir::Complex64Type>()) {
    return phi::DataType::COMPLEX64;
  } else if (dtype.isa<pir::Complex128Type>()) {
    return phi::DataType::COMPLEX128;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported ir data type when casting it into "
        "phi data type."));
  }
}

// use phi::DataType::INT32 for IndexType from builtin type to phi::DataType,
// but only use INT32 not IndexType from phi::DataType type to builtin type.
static inline pir::Type TransToIrDataType(phi::DataType dtype,
                                          pir::IrContext* ctx = nullptr) {
  if (ctx == nullptr) {
    ctx = pir::IrContext::Instance();
  }
  switch (dtype) {
    case phi::DataType::BFLOAT16:
      return pir::BFloat16Type::get(ctx);
    case phi::DataType::FLOAT16:
      return pir::Float16Type::get(ctx);
    case phi::DataType::FLOAT32:
      return pir::Float32Type::get(ctx);
    case phi::DataType::FLOAT64:
      return pir::Float64Type::get(ctx);
    case phi::DataType::UINT8:
      return pir::UInt8Type::get(ctx);
    case phi::DataType::INT8:
      return pir::Int8Type::get(ctx);
    case phi::DataType::INT16:
      return pir::Int16Type::get(ctx);
    case phi::DataType::INT32:
      return pir::Int32Type::get(ctx);
    case phi::DataType::INT64:
      return pir::Int64Type::get(ctx);
    case phi::DataType::BOOL:
      return pir::BoolType::get(ctx);
    case phi::DataType::COMPLEX64:
      return pir::Complex64Type::get(ctx);
    case phi::DataType::COMPLEX128:
      return pir::Complex128Type::get(ctx);
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported phi data type `%s` when casting it into "
          "ir data type.",
          dtype));
  }
}

static inline pir::Attribute TransToIrAttribute(phi::Scalar scalar,
                                                pir::IrContext* ctx = nullptr) {
  if (ctx == nullptr) {
    ctx = pir::IrContext::Instance();
  }
  switch (scalar.dtype()) {
    case phi::DataType::FLOAT32:
      return pir::FloatAttribute::get(ctx, scalar.to<float>());
    case phi::DataType::FLOAT64:
      return pir::DoubleAttribute::get(ctx, scalar.to<double>());
    case phi::DataType::INT32:
      return pir::Int32Attribute::get(ctx, scalar.to<int32_t>());
    case phi::DataType::INT64:
      return pir::Int64Attribute::get(ctx, scalar.to<int64_t>());
    case phi::DataType::BOOL:
      return pir::BoolAttribute::get(ctx, scalar.to<bool>());
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported phi data type `%s` when casting it into "
          "ir attribute.",
          scalar.dtype()));
  }
}

inline DataType VarTypeToDataType(
    ::paddle::framework::proto::VarType_Type var_type) {
  switch (var_type) {
    case paddle::framework::proto::VarType_Type::VarType_Type_BOOL:
      return DataType::BOOL;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT16:
      return DataType::INT16;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT32:
      return DataType::INT32;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT64:
      return DataType::INT64;
    case paddle::framework::proto::VarType_Type::VarType_Type_FP16:
      return DataType::FLOAT16;
    case paddle::framework::proto::VarType_Type::VarType_Type_FP32:
      return DataType::FLOAT32;
    case paddle::framework::proto::VarType_Type::VarType_Type_FP64:
      return DataType::FLOAT64;
    case paddle::framework::proto::VarType_Type::VarType_Type_SIZE_T:
      return DataType::UINT64;
    case paddle::framework::proto::VarType_Type::VarType_Type_UINT8:
      return DataType::UINT8;
    case paddle::framework::proto::VarType_Type::VarType_Type_INT8:
      return DataType::INT8;
    case paddle::framework::proto::VarType_Type::VarType_Type_BF16:
      return DataType::BFLOAT16;
    case paddle::framework::proto::VarType_Type::VarType_Type_COMPLEX64:
      return DataType::COMPLEX64;
    case paddle::framework::proto::VarType_Type::VarType_Type_COMPLEX128:
      return DataType::COMPLEX128;
    case paddle::framework::proto::VarType_Type::VarType_Type_PSTRING:
      return DataType::PSTRING;
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported proto::VarType_Type `%s` when casting it into DataType.",
          var_type));
  }
}

VariantType GetAttributeData(const pir::Attribute& attr);

bool IsLegacyOp(const std::string& name);

bool IsEmptyOpResult(const pir::OpResult& op_result);

}  // namespace dialect
}  // namespace paddle
