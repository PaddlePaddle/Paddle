// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/macros.h"

#include "paddle/phi/common/reduce_type.h"

namespace phi {
namespace ccl {
typedef void* CCLComm;
typedef std::vector<uint8_t> CCLRootId;

enum CCLReduceOp { SUM = 0, AVG, MAX, MIN, PRODUCT };
enum CCLDataType {
  CCL_DATA_TYPE_FP64 = 0,
  CCL_DATA_TYPE_FP32,
  CCL_DATA_TYPE_FP16,
  CCL_DATA_TYPE_INT64,
  CCL_DATA_TYPE_INT32,
  CCL_DATA_TYPE_INT16,
  CCL_DATA_TYPE_INT8,
  CCL_DATA_TYPE_UINT8
};

inline CCLReduceOp ToXCCLReduceOp(int reduce_type) {
  phi::ccl::CCLReduceOp red_type = phi::ccl::CCLReduceOp::SUM;
  switch (static_cast<phi::ReduceType>(reduce_type)) {
    case phi::ReduceType::kRedSum:
      red_type = phi::ccl::CCLReduceOp::SUM;
      break;
    case phi::ReduceType::kRedMax:
      red_type = phi::ccl::CCLReduceOp::MAX;
      break;
    case phi::ReduceType::kRedMin:
      red_type = phi::ccl::CCLReduceOp::MIN;
      break;
    case phi::ReduceType::kRedProd:
      red_type = phi::ccl::CCLReduceOp::PRODUCT;
      break;
    case phi::ReduceType::kRedAvg:
      red_type = phi::ccl::CCLReduceOp::AVG;
      break;
    default:
      PADDLE_THROW(
          errors::Unavailable("Unsuppored reduce type. Reduce type must be one "
                              "of SUM, MAX, MIN, PRODUCT and AVG."));
  }
  return red_type;
}

inline CCLDataType ToCCLDataType(phi::DataType type) {
  if (type == phi::DataType::FLOAT64) {
    return CCL_DATA_TYPE_FP64;
  } else if (type == phi::DataType::FLOAT32) {
    return CCL_DATA_TYPE_FP32;
  } else if (type == phi::DataType::FLOAT16) {
    return CCL_DATA_TYPE_FP16;
  } else if (type == phi::DataType::INT64) {
    return CCL_DATA_TYPE_INT64;
  } else if (type == phi::DataType::INT32) {
    return CCL_DATA_TYPE_INT32;
  } else if (type == phi::DataType::INT8) {
    return CCL_DATA_TYPE_INT8;
  } else if (type == phi::DataType::UINT8) {
    return CCL_DATA_TYPE_UINT8;
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("This datatype %s in CCL is not supported.",
                                   phi::DataTypeToString(type)));
  }
}

inline phi::DataType ToPhiDataType(CCLDataType type) {
  if (type == CCLDataType::CCL_DATA_TYPE_FP64) {
    return phi::DataType::FLOAT64;
  } else if (type == CCLDataType::CCL_DATA_TYPE_FP32) {
    return phi::DataType::FLOAT32;
  } else if (type == CCLDataType::CCL_DATA_TYPE_FP16) {
    return phi::DataType::FLOAT16;
  } else if (type == CCLDataType::CCL_DATA_TYPE_INT64) {
    return phi::DataType::INT64;
  } else if (type == CCLDataType::CCL_DATA_TYPE_INT32) {
    return phi::DataType::INT32;
  } else if (type == CCLDataType::CCL_DATA_TYPE_INT8) {
    return phi::DataType::INT8;
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("This datatype in CCL is not supported."));
  }
}

inline std::string SerializeXCCLUniqueId(const phi::ccl::CCLRootId& ccl_id) {
  const uint8_t* bytes = ccl_id.data();
  std::ostringstream oss;
  for (size_t i = 0; i < ccl_id.size(); ++i) {
    oss << std::hex << static_cast<int>(bytes[i]);
  }
  return oss.str();
}

}  // namespace ccl
}  // namespace phi
