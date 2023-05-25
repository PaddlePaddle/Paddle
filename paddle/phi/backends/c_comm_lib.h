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
  CCL_DATA_TYPE_INT8
};

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
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("This datatype in CCL is not supported."));
  }
}

}  // namespace ccl
}  // namespace phi
