// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>

#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

class CustomDeviceFuncBase {
 public:
  virtual ~CustomDeviceFuncBase() = default;

  virtual void CustomCastDataType(const CustomContext& dev_ctx,
                                  const void* in,
                                  void* out,
                                  int64_t numel,
                                  DataType in_dtype,
                                  DataType out_dtype) const {
    PADDLE_THROW(common::errors::Unimplemented(
        "Custom Transform is not available in this build."));
  }
};

// Factory function declaration
std::unique_ptr<CustomDeviceFuncBase> CreateCustomDeviceFunc();

}  // namespace phi
