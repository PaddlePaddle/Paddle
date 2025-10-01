// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <utils/macros.h>
#include "paddle/phi/common/data_type.h"

namespace compat {
inline phi::DataType _PD_AtenScalarTypeToPhiDataType(c10::ScalarType dtype) {
  switch (dtype) {
#define DEFINE_ST_TO_DT_CASE_(_1, _dt, _st) \
  case c10::ScalarType::_st:                \
    return phi::DataType::_dt;
    FOREACH_PADDLE_AND_TORCH_DTYPES(DEFINE_ST_TO_DT_CASE_)
#undef DEFINE_ST_TO_DT_CASE_
    case c10::ScalarType::Undefined:
      return phi::DataType::UNDEFINED;
    default:
      UNSUPPORTED_FEATURE_IN_PADDLE("Unsupported ScalarType")
      return phi::DataType::UNDEFINED;  // to avoid compile warning
  }
}

inline c10::ScalarType _PD_PhiDataTypeToAtenScalarType(phi::DataType dtype) {
  switch (dtype) {
#define DEFINE_DT_TO_ST_CASE_(_1, _dt, _st) \
  case phi::DataType::_dt:                  \
    return c10::ScalarType::_st;
    FOREACH_PADDLE_AND_TORCH_DTYPES(DEFINE_DT_TO_ST_CASE_)
#undef DEFINE_DT_TO_ST_CASE_
    case phi::DataType::UNDEFINED:
      return c10::ScalarType::Undefined;
    default:
      UNSUPPORTED_FEATURE_IN_PADDLE("Unsupported DataType")
      return c10::ScalarType::Undefined;  // to avoid compile warning
  }
}

}  // namespace compat
