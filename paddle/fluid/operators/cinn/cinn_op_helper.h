// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/operator.h"

// We define some common names or utility functions
// for operators related to cinn in this file
namespace paddle::operators {

// input params, output params and attributes
constexpr char kX[] = "X";
constexpr char kNoNeedBufferX[] = "NoNeedBufferX";
constexpr char kOutputs[] = "Out";
constexpr char kCompilationKey[] = "compilation_key";
constexpr char kCachedIndex[] = "cached_index";
constexpr char kInstructionIndex[] = "instruction_index";

// utility functions
namespace details {

template <typename DeviceContext>
void* GetStream(const framework::ExecutionContext& ctx) {
  return nullptr;
}

#ifdef PADDLE_WITH_CUDA
template <>
void* GetStream<platform::CUDADeviceContext>(
    const framework::ExecutionContext& ctx);
#endif

}  // namespace details
}  // namespace paddle::operators
