// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/data_value.h"
#include "paddle/phi/common/scalar.h"

namespace ap::paddle {

struct ScalarHelper {
  adt::Result<phi::Scalar> ConvertFromDataType(
      const axpr::DataValue& data_val) {
    using RetT = adt::Result<phi::Scalar>;
    return data_val.Match(
        [&](double c) -> RetT { return phi::Scalar(c); },
        [&](float c) -> RetT { return phi::Scalar(c); },
        [&](axpr::float16 c) -> RetT { return phi::Scalar(c); },
        [&](axpr::bfloat16 c) -> RetT { return phi::Scalar(c); },
        [&](int64_t c) -> RetT { return phi::Scalar(c); },
        [&](int32_t c) -> RetT { return phi::Scalar(c); },
        [&](int16_t c) -> RetT { return phi::Scalar(c); },
        [&](int8_t c) -> RetT { return phi::Scalar(c); },
        [&](uint64_t c) -> RetT { return phi::Scalar(c); },
        [&](uint32_t c) -> RetT { return phi::Scalar(c); },
        [&](uint16_t c) -> RetT { return phi::Scalar(c); },
        [&](uint8_t c) -> RetT { return phi::Scalar(c); },
        [&](bool c) -> RetT { return phi::Scalar(c); },
        [&](const axpr::complex64& c) -> RetT { return phi::Scalar(c); },
        [&](const axpr::complex128& c) -> RetT { return phi::Scalar(c); },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() + "ConvertFromDataType(): can not convert from " +
              data_val.GetType().Name() + " to phi::Scalar"};
        });
  }

  adt::Result<axpr::DataValue> ConvertToDataValue(const phi::Scalar& scalar) {
    switch (scalar.dtype()) {
      case phi::DataType::FLOAT32:
        return axpr::DataValue(scalar.to<float>());
      case phi::DataType::FLOAT64:
        return axpr::DataValue(scalar.to<double>());
      case phi::DataType::FLOAT16:
        return axpr::DataValue(scalar.to<phi::float16>());
      case phi::DataType::BFLOAT16:
        return axpr::DataValue(scalar.to<phi::bfloat16>());
      case phi::DataType::INT32:
        return axpr::DataValue(scalar.to<int32_t>());
      case phi::DataType::INT64:
        return axpr::DataValue(scalar.to<int64_t>());
      case phi::DataType::INT16:
        return axpr::DataValue(scalar.to<int16_t>());
      case phi::DataType::INT8:
        return axpr::DataValue(scalar.to<int8_t>());
      case phi::DataType::UINT64:
        return axpr::DataValue(scalar.to<uint64_t>());
      case phi::DataType::UINT32:
        return axpr::DataValue(scalar.to<uint32_t>());
      case phi::DataType::UINT16:
        return axpr::DataValue(scalar.to<uint16_t>());
      case phi::DataType::UINT8:
        return axpr::DataValue(scalar.to<uint8_t>());
      case phi::DataType::BOOL:
        return axpr::DataValue(scalar.to<bool>());
      case phi::DataType::COMPLEX64:
        return axpr::DataValue(scalar.to<phi::complex64>());
      case phi::DataType::COMPLEX128:
        return axpr::DataValue(scalar.to<phi::complex128>());
      default:
        std::ostringstream ss;
        ss << scalar.dtype();
        return adt::errors::TypeError{std::string() +
                                      "Invalid enum scalar data type `" +
                                      ss.str() + "`."};
    }
  }
};

}  // namespace ap::paddle
