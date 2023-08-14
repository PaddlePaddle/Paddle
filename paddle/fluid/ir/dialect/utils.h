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
  } else if (dtype.isa<ir::IndexType>()) {
    return phi::DataType::INT32;
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

// use phi::DataType::INT32 for IndexType from builtin type to phi::DataType,
// but only use INT32 not IndexType from phi::DataType type to builtin type.
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

VariantType GetAttributeData(const ir::Attribute& attr);

}  // namespace dialect
}  // namespace paddle
