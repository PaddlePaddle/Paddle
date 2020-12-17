// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <string>
#include <unordered_map>

#include "paddle/fluid/platform/errors.h"
#include "xpu/api.h"
#include "xpu/refactor/fusion.h"
#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"
#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"

namespace xpu = baidu::xpu::api;

class XPUActHelper {
 public:
  // Convert string to activation type in xpu
  static xpu::Activation_t ConvertToXpuActType(
      const std::string& act_type_str) {
    static std::unordered_map<std::string, xpu::Activation_t> str2act = {
        {"linear", xpu::Activation_t::LINEAR},
        {"relu", xpu::Activation_t::RELU},
        {"sigmoid", xpu::Activation_t::SIGMOID},
        {"tanh", xpu::Activation_t::TANH},
        {"gelu", xpu::Activation_t::GELU},
        {"leaky_relu", xpu::Activation_t::LEAKY_RELU},
        {"sqrt", xpu::Activation_t::SQRT},
        {"square", xpu::Activation_t::SQUARE}};

    auto res = str2act.find(act_type_str);
    PADDLE_ENFORCE_NE(res, str2act.end(),
                      paddle::platform::errors::InvalidArgument(
                          "Invalid activation type(%s) in XPU", act_type_str));
    return res->second;
  }
};

static std::map<int, std::string> XPUAPIErrorMsg = {
    {xpu::Error_t::SUCCESS, "xpu api success"},
    {xpu::Error_t::INVALID_PARAM, "xpu api invalid param"},
    {xpu::Error_t::RUNTIME_ERROR, "xpu api runtime error"},
    {xpu::Error_t::NO_ENOUGH_WORKSPACE, "xpu api no enough workspace"}};

#endif
