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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

template <typename T,
          typename MT,
          std::enable_if_t<std::is_same<T, float>::value, bool> = true>
HOSTDEVICE bool NeedPrint(MT max_value, MT min_value, int check_nan_inf_level) {
  if (check_nan_inf_level >= 3) {
    return true;
  } else if (check_nan_inf_level >= 2) {
    MT fp16_max =
        static_cast<MT>(std::numeric_limits<phi::dtype::float16>::max());
    return max_value > fp16_max || min_value < -fp16_max;
  }
  return false;
}

template <typename T,
          typename MT,
          std::enable_if_t<!std::is_same<T, float>::value, bool> = true>
HOSTDEVICE bool NeedPrint(MT max_value, MT min_value, int check_nan_inf_level) {
  if (check_nan_inf_level >= 3) {
    return true;
  }
  return false;
}

template <typename T, typename MT>
HOSTDEVICE void PrintForDifferentLevel(const char* debug_info,
                                       int64_t numel,
                                       int64_t num_nan,
                                       int64_t num_inf,
                                       int64_t num_zero,
                                       MT max_value,
                                       MT min_value,
                                       MT mean_value,
                                       int check_nan_inf_level) {
  if (num_nan > 0 || num_inf > 0) {
    printf(
        "[PRECISION] [ERROR] in %s, numel=%lld, num_nan=%lld, "
        "num_inf=%lld, num_zero=%lld, max=%e, min=%e, mean=%e\n",
        debug_info,
        static_cast<long long>(numel),     // NOLINT
        static_cast<long long>(num_nan),   // NOLINT
        static_cast<long long>(num_inf),   // NOLINT
        static_cast<long long>(num_zero),  // NOLINT
        static_cast<float>(max_value),
        static_cast<float>(min_value),
        static_cast<float>(mean_value));
    if (check_nan_inf_level == 0) {
#if defined(__NVCC__) || defined(__HIPCC__)
      PADDLE_ENFORCE(false,
                     "There are NAN or INF (num_nan=%ld, num_inf=%lld, "
                     "num_zero=%lld) in %s.",
                     static_cast<long long>(num_nan),   // NOLINT
                     static_cast<long long>(num_inf),   // NOLINT
                     static_cast<long long>(num_zero),  // NOLINT
                     debug_info);
#else
      PADDLE_THROW(phi::errors::PreconditionNotMet(
          "There are NAN or INF (num_nan=%lld, num_inf=%lld, num_zero=%lld) in "
          "%s.",
          static_cast<long long>(num_nan),   // NOLINT
          static_cast<long long>(num_inf),   // NOLINT
          static_cast<long long>(num_zero),  // NOLINT
          debug_info));
#endif
    }
  } else if (NeedPrint<T, MT>(max_value, min_value, check_nan_inf_level)) {
    printf("[PRECISION] in %s, numel=%lld, max=%e, min=%e, mean=%e\n",
           debug_info,
           static_cast<long long>(numel),  // NOLINT
           static_cast<float>(max_value),
           static_cast<float>(min_value),
           static_cast<float>(mean_value));
  }
}

template <typename T>
inline std::string GetCpuHintString(const std::string& op_type,
                                    const std::string& var_name,
                                    const phi::Place& place,
                                    int device_id = -1) {
  std::string dtype_str;
  phi::DataType dtype = phi::CppTypeToDataType<T>::Type();
  if (dtype == DataType::FLOAT32) {
    dtype_str = "fp32";
  } else if (dtype == DataType::FLOAT64) {
    dtype_str = "fp64";
  } else if (dtype == DataType::FLOAT16) {
    dtype_str = "fp16";
  } else if (dtype == DataType::BFLOAT16) {
    dtype_str = "bf16";
  }

  std::stringstream ss;
  if (place.GetType() == phi::AllocationType::GPU) {
    ss << "[device=gpu:" << device_id << ", ";
  } else {
    ss << "[device=cpu, ";
  }
  ss << "op=" << op_type << ", tensor=" << var_name << ", dtype=" << dtype_str
     << "]";
  return ss.str();
}

}  // namespace funcs
}  // namespace phi
