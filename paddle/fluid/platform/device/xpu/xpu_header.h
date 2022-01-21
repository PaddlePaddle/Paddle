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
#include <map>
#include <string>
#include <unordered_map>

#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

#include "xpu/runtime.h"
#include "xpu/runtime_ex.h"
#include "xpu/xdnn.h"

namespace xpu = baidu::xpu::api;

static std::map<int, std::string> XPUAPIErrorMsg = {
    {xpu::Error_t::SUCCESS, "xpu api success"},
    {xpu::Error_t::INVALID_PARAM, "xpu api invalid param"},
    {xpu::Error_t::RUNTIME_ERROR, "xpu api runtime error"},
    {xpu::Error_t::NO_ENOUGH_WORKSPACE, "xpu api no enough workspace"}};

template <typename T>
class XPUTypeTrait {
 public:
  using Type = T;
};

template <>
class XPUTypeTrait<paddle::platform::float16> {
 public:
  using Type = float16;
};

template <>
class XPUTypeTrait<paddle::platform::bfloat16> {
 public:
  using Type = bfloat16;
};

#endif
