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

#include "paddle/phi/backends/custom/custom_device_func.h"

#if defined(PADDLE_WITH_CUSTOM_DEVICE_SUB_BUILD)
#include "common/custom_device_func.h"
#endif

namespace phi {

std::unique_ptr<CustomDeviceFuncBase> CreateCustomDeviceFunc() {
#if defined(PADDLE_WITH_CUSTOM_DEVICE_SUB_BUILD)
  return std::make_unique<CustomDeviceFunc>();
#else
  return std::make_unique<CustomDeviceFuncBase>();
#endif
}

}  // namespace phi
