// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace details {

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
                                       int has_nan,
                                       int has_inf,
                                       MT max_value,
                                       MT min_value,
                                       MT mean_value,
                                       int check_nan_inf_level) {
  if (has_nan || has_inf) {
    printf(
        "[PRECISION] [ERROR] in %s, numel=%ld, find_nan=%d, "
        "find_inf=%d, "
        "max=%e, min=%e, mean=%e\n",
        debug_info,
        numel,
        has_nan,
        has_inf,
        static_cast<float>(max_value),
        static_cast<float>(min_value),
        static_cast<float>(mean_value));
    if (check_nan_inf_level == 0) {
#if defined(__NVCC__) || defined(__HIPCC__)
      PADDLE_ENFORCE(false, "Find nan or inf in %s.", debug_info);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "Find nan or inf in %s.", debug_info));
#endif
    }
  } else if (NeedPrint<T, MT>(max_value, min_value, check_nan_inf_level)) {
    printf("[PRECISION] in %s, numel=%ld, max=%e, min=%e, mean=%e\n",
           debug_info,
           numel,
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
  std::string dtype_str = DataTypeToString(DataTypeTrait<T>::DataType());
  if (dtype_str == "::paddle::platform::float16") {
    dtype_str = "float16";
  }

  std::stringstream ss;
  if (platform::is_gpu_place(place)) {
    ss << "[device=gpu:" << device_id << ", ";
  } else {
    ss << "[device=cpu, ";
  }
  ss << "op=" << op_type << ", tensor=" << var_name << ", dtype=" << dtype_str
     << "]";
  return ss.str();
}

template <typename DeviceContext>
struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string& op_type,
                       const std::string& var_name,
                       const phi::DenseTensor& tensor,
                       const platform::Place& place)
      : op_type_(op_type),
        var_name_(var_name),
        tensor_(tensor),
        place_(place) {}

  template <typename T>
  void apply(
      typename std::enable_if<std::is_integral<T>::value>::type* = 0) const {
    VLOG(10) << var_name_ << " need not to check, it's type is not float point";
  }

  template <typename T>
  void apply(
      typename std::enable_if<
          std::is_floating_point<T>::value ||
          std::is_same<T, ::paddle::platform::complex<float>>::value ||
          std::is_same<T, ::paddle::platform::complex<double>>::value>::type* =
          0) const;

  std::string op_type_;
  std::string var_name_;
  const phi::DenseTensor& tensor_;
  const platform::Place& place_;
};

template <typename DeviceContext>
void tensor_check(const std::string& op_type,
                  const std::string& var_name,
                  const phi::DenseTensor& tensor,
                  const platform::Place& place);

}  // namespace details
}  // namespace framework
}  // namespace paddle
