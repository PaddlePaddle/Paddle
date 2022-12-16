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

#include "paddle/fluid/framework/paddle2cinn/transform_type.h"

#include "cinn/common/type.h"
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle::framework::paddle2cinn {

::phi::DataType TransToPaddleDataType(const ::cinn::common::Type& type) {
#define SET_TYPE_CASE_ITEM(cinn_common_type, pd_type) \
  if (type == ::cinn::common::cinn_common_type()) {   \
    return ::phi::DataType::pd_type;                  \
  }

  SET_TYPE_CASE_ITEM(Bool, BOOL)
  SET_TYPE_CASE_ITEM(I8, INT8)
  SET_TYPE_CASE_ITEM(I16, INT16)
  SET_TYPE_CASE_ITEM(I32, INT32)
  SET_TYPE_CASE_ITEM(I64, INT64)
  SET_TYPE_CASE_ITEM(UI8, UINT8)
  SET_TYPE_CASE_ITEM(UI16, UINT16)
  SET_TYPE_CASE_ITEM(UI32, UINT32)
  SET_TYPE_CASE_ITEM(UI64, UINT64)
  SET_TYPE_CASE_ITEM(F16, FLOAT16)
  SET_TYPE_CASE_ITEM(F32, FLOAT32)
  SET_TYPE_CASE_ITEM(F64, FLOAT64)

  PADDLE_THROW(
      platform::errors::Unimplemented("Type(%s) not supported yet", type));
  return ::phi::DataType::UNDEFINED;
#undef SET_TYPE_CASE_ITEM
}

::phi::DataType TransToPaddleDataType(const cinn_type_t& type) {
#define SET_TYPE_CASE_ITEM(cinn_runtime_type, pd_type) \
  if (type == cinn_runtime_type()) {                   \
    return ::phi::DataType::pd_type;                   \
  }

  SET_TYPE_CASE_ITEM(cinn_bool_t, BOOL)
  SET_TYPE_CASE_ITEM(cinn_int8_t, INT8)
  SET_TYPE_CASE_ITEM(cinn_int32_t, INT32)
  SET_TYPE_CASE_ITEM(cinn_int64_t, INT64)
  SET_TYPE_CASE_ITEM(cinn_uint32_t, UINT32)
  SET_TYPE_CASE_ITEM(cinn_uint64_t, UINT64)
  SET_TYPE_CASE_ITEM(cinn_float32_t, FLOAT32)
  SET_TYPE_CASE_ITEM(cinn_float64_t, FLOAT64)
#ifdef CINN_COMMON_FLOAT16_H
  SET_TYPE_CASE_ITEM(cinn_float16_t, FLOAT16)
#endif  // CINN_COMMON_FLOAT16_H

  PADDLE_THROW(platform::errors::Unimplemented("Input type not supported yet"));
  return ::phi::DataType::UNDEFINED;
#undef SET_TYPE_CASE_ITEM
}

}  // namespace paddle::framework::paddle2cinn
