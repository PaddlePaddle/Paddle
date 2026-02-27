/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#ifdef PADDLE_WITH_XPU
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/phi/common/data_type.h"

namespace phi {
namespace backends {
namespace xpu {
constexpr DataType FLOAT32 = DataType::FLOAT32;
constexpr DataType FLOAT64 = DataType::FLOAT64;
constexpr DataType FLOAT16 = DataType::FLOAT16;
constexpr DataType BFLOAT16 = DataType::BFLOAT16;
constexpr DataType INT32 = DataType::INT32;
constexpr DataType INT64 = DataType::INT64;
constexpr DataType UINT8 = DataType::UINT8;
constexpr DataType INT16 = DataType::INT16;
constexpr DataType INT8 = DataType::INT8;
constexpr DataType BOOL = DataType::BOOL;
constexpr DataType COMPLEX64 = DataType::COMPLEX64;
constexpr DataType COMPLEX128 = DataType::COMPLEX128;
using XPUKernelSet = std::unordered_set<phi::DataType>;
using XPUOpMap = std::unordered_map<std::string, XPUKernelSet>;

XPUOpMap& get_kl1_ops();
XPUOpMap& get_kl2_ops();
XPUOpMap& get_kl3_ops();

#ifdef PADDLE_WITH_XPU_KP
bool is_xpu_kp_support_op(const std::string& fluid_op_name,
                          const phi::DataType type);
#endif

bool is_in_xpu_black_list(const std::string& fluid_op_name);
bool is_xpu_support_op(const std::string& fluid_op_name,
                       const phi::DataType type);

}  // namespace xpu
}  // namespace backends
}  // namespace phi
#endif
