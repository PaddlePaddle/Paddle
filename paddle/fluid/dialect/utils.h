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

#include "paddle/fluid/dialect/pd_type_storage.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace paddle {
namespace dialect {
// TODO(zhangbo): The builtin type needs to cover all data types of
// phi::DataType.
inline phi::DataType TransToPhiDataType(ir::Type dtype) {
  if (dtype.isa<ir::Float16Type>()) {
    return phi::DataType::FLOAT16;
  } else if (dtype.isa<ir::Float32Type>()) {
    return phi::DataType::FLOAT32;
  } else if (dtype.isa<ir::Float64Type>()) {
    return phi::DataType::FLOAT64;
  } else if (dtype.isa<ir::Int16Type>()) {
    return phi::DataType::INT16;
  } else if (dtype.isa<ir::Int32Type>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<ir::Int64Type>()) {
    return phi::DataType::INT64;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported ir data type when casting it into "
        "phi data type."));
  }
}

inline ir::Type TransToIrDataType(phi::DataType dtype,
                                  ir::IrContext *ctx = nullptr) {
  if (ctx == nullptr) {
    ctx = ir::IrContext::Instance();
  }
  switch (dtype) {
    case phi::DataType::FLOAT16:
      return ir::Float16Type::get(ctx);
    case phi::DataType::FLOAT32:
      return ir::Float32Type::get(ctx);
    case phi::DataType::FLOAT64:
      return ir::Float64Type::get(ctx);
    case phi::DataType::INT16:
      return ir::Int16Type::get(ctx);
    case phi::DataType::INT32:
      return ir::Int32Type::get(ctx);
    case phi::DataType::INT64:
      return ir::Int64Type::get(ctx);
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Unsupported phi data type `%s` when casting it into "
          "ir data type.",
          dtype));
  }
}

// inline phi::DataLayout TransToPhiDataLayout(
//     DenseTensorTypeStorage::DataLayout data_layout) {
//   switch (data_layout) {
//     case DenseTensorTypeStorage::DataLayout::NHWC:
//       return phi::DataLayout::NHWC;
//     case DenseTensorTypeStorage::DataLayout::NCHW:
//       return phi::DataLayout::NCHW;
//     case DenseTensorTypeStorage::DataLayout::NCDHW:
//       return phi::DataLayout::NCDHW;
//     case DenseTensorTypeStorage::DataLayout::NDHWC:
//       return phi::DataLayout::NDHWC;
//     case DenseTensorTypeStorage::DataLayout::ONEDNN:
//       return phi::DataLayout::ONEDNN;
//     case DenseTensorTypeStorage::DataLayout::SPARSE_COO:
//       return phi::DataLayout::SPARSE_COO;
//     case DenseTensorTypeStorage::DataLayout::SPARSE_CSR:
//       return phi::DataLayout::SPARSE_CSR;
//     case DenseTensorTypeStorage::DataLayout::PSTRING_UNION:
//       return phi::DataLayout::PSTRING_UNION;
//     case DenseTensorTypeStorage::DataLayout::NUM_DATA_LAYOUTS:
//       return phi::DataLayout::NUM_DATA_LAYOUTS;
//     case DenseTensorTypeStorage::DataLayout::ALL_LAYOUT:
//       return phi::DataLayout::ALL_LAYOUT;
//     default:
//       PADDLE_THROW(phi::errors::Unimplemented(
//           "Unsupported ir data layout `%s` when casting it into "
//           "phi data type.",
//           static_cast<int>(data_layout)));
//   }
// }

// inline DenseTensorTypeStorage::DataLayout TransToIrDataLayout(
//     phi::DataLayout data_layout) {
//   switch (data_layout) {
//     case phi::DataLayout::NHWC:
//       return DenseTensorTypeStorage::DataLayout::NHWC;
//     case phi::DataLayout::NCHW:
//       return DenseTensorTypeStorage::DataLayout::NCHW;
//     case phi::DataLayout::NCDHW:
//       return DenseTensorTypeStorage::DataLayout::NCDHW;
//     case phi::DataLayout::NDHWC:
//       return DenseTensorTypeStorage::DataLayout::NDHWC;
//     case phi::DataLayout::ONEDNN:
//       return DenseTensorTypeStorage::DataLayout::ONEDNN;
//     case phi::DataLayout::SPARSE_COO:
//       return DenseTensorTypeStorage::DataLayout::SPARSE_COO;
//     case phi::DataLayout::SPARSE_CSR:
//       return DenseTensorTypeStorage::DataLayout::SPARSE_CSR;
//     case phi::DataLayout::PSTRING_UNION:
//       return DenseTensorTypeStorage::DataLayout::PSTRING_UNION;
//     case phi::DataLayout::NUM_DATA_LAYOUTS:
//       return DenseTensorTypeStorage::DataLayout::NUM_DATA_LAYOUTS;
//     case phi::DataLayout::ALL_LAYOUT:
//       return DenseTensorTypeStorage::DataLayout::ALL_LAYOUT;
//     default:
//       PADDLE_THROW(phi::errors::Unimplemented(
//           "Unsupported phi data layout `%s` when casting it into "
//           "ir data type.",
//           static_cast<int>(data_layout)));
//   }
// }

// inline phi::DenseTensorMeta TransToDenseTensorMeta(
//     paddle::dialect::DenseTensorType type) {
//   return phi::DenseTensorMeta(TransToPhiDataType(type.dtype()),
//                               type.dim(),
//                               type.data_layout(),
//                               type.lod(),
//                               type.offset());
// }

struct OpInputInfo {
  std::string name;
  std::string type_name;
  bool optional = false;
  bool no_need_buffer = false;
  OpInputInfo(std::string name,
              std::string type_name,
              bool optional,
              bool no_need_buffer)
      : name(name),
        type_name(type_name),
        optional(optional),
        no_need_buffer(no_need_buffer) {}
};

struct OpOutputInfo {
  std::string name;
  std::string type_name;
  bool optional = false;
  bool intermediate = false;
  OpOutputInfo(std::string name,
               std::string type_name,
               bool optional,
               bool intermediate)
      : name(name),
        type_name(type_name),
        optional(optional),
        intermediate(intermediate) {}
};

struct OpAttributeInfo {
  std::string name;
  std::string type_name;
  std::string data_type;
  OpAttributeInfo(std::string name,
                  std::string type_name,
                  std::string data_type)
      : name(name), type_name(type_name), data_type(data_type) {}
};

struct OpRunTimeInfo {
  std::string infer_meta_func;
  std::vector<std::string> infer_meta_param;
  std::vector<std::string> kernel_func;
  std::vector<std::string> kernel_param;
  OpRunTimeInfo(std::string infer_meta_func,
                std::vector<std::string> infer_meta_param,
                std::vector<std::string> kernel_func,
                std::vector<std::string> kernel_param)
      : infer_meta_func(infer_meta_func),
        infer_meta_param(infer_meta_param),
        kernel_func(kernel_func),
        kernel_param(kernel_param) {}
};

}  // namespace dialect
}  // namespace paddle
