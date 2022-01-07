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

// Note: Some scenarios need to include all types of Context declarations.
// In order to avoid including the header files of each backend in turn,
// add this header file
// Note: Limit the entry of DeviceContext to backends to avoid multiple include
// path replacement after implementing pten DeviceContext

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/backends/xpu/xpu_context.h"

namespace pten {
using DeviceContext = paddle::platform::DeviceContext;
using DeviceContextPool = paddle::platform::DeviceContextPool;
}  // namespace pten
