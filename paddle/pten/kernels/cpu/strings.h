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

#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/string_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"

namespace pten {

using CPUContext = paddle::platform::CPUDeviceContext;
template <typename AsciiCoverter>
void StringCaseConvert(const CPUContext& dev_ctx,
                       const StringTensor& x,
                       const std::string& encoding,
                       StringTensor* out);

void StringLower(const CPUContext& dev_ctx,
                 const StringTensor& x,
                 const std::string& encoding,
                 StringTensor* out);

void StringUpper(const CPUContext& dev_ctx,
                 const StringTensor& x,
                 const std::string& encoding,
                 StringTensor* out);

}  // namespace pten
