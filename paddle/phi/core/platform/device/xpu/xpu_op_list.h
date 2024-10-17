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

#ifdef PADDLE_WITH_XPU
#include <string>
#include <unordered_map>

#include "paddle/phi/backends/xpu/xpu_op_list.h"
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/xpu_op_kpfirst_list.h"
#endif
#include "paddle/phi/core/framework/framework.pb.h"

namespace paddle {
namespace platform {

using phi::backends::xpu::is_in_xpu_black_list;
using phi::backends::xpu::is_xpu_support_op;
using vartype = paddle::framework::proto::VarType;
using XPUOpListMap =
    std::unordered_map<std::string, std::vector<vartype::Type>>;

#ifdef PADDLE_WITH_XPU_KP
using phi::backends::xpu::is_xpu_kp_support_op;
std::vector<vartype::Type> get_xpu_kp_op_support_type(
    const std::string& op_name, phi::backends::xpu::XPUVersion version);
bool is_in_xpu_kpwhite_list(const std::string& op_name);
#endif

std::vector<vartype::Type> get_xpu_op_support_type(
    const std::string& op_name, phi::backends::xpu::XPUVersion version);
XPUOpListMap get_xpu_op_list(phi::backends::xpu::XPUVersion version);

}  // namespace platform
}  // namespace paddle
#endif
