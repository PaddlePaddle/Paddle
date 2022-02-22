/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_kernel_type.h"

namespace paddle {
namespace platform {

using pOpKernelType = paddle::framework::OpKernelType;
using vartype = paddle::framework::proto::VarType;
using XPUOpListMap =
    std::unordered_map<std::string, std::vector<vartype::Type>>;

bool is_xpu_support_op(const std::string& op_name, const pOpKernelType& type);
bool is_in_xpu_black_list(const std::string& op_name);

#ifdef PADDLE_WITH_XPU_KP
bool is_xpu_kp_support_op(const std::string& op_name,
                          const pOpKernelType& type);
bool is_in_xpu_kpwhite_list(const std::string& op_name);
#endif

std::vector<vartype::Type> get_xpu_op_support_type(
    const std::string& op_name, phi::backends::xpu::XPUVersion version);
XPUOpListMap get_xpu_op_list(phi::backends::xpu::XPUVersion version);

}  // namespace platform
}  // namespace paddle
#endif
