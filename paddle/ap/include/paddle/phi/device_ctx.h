// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <any>
#include "paddle/ap/include/kernel_dispatch/device_ctx.h"

namespace ap::paddle {

template <typename PhiDeviceCtx>
class DeviceCtx : public kernel_dispatch::DeviceCtxImpl {
 private:
  const PhiDeviceCtx* phi_device_ctx_;
  std::any stream_;

 public:
  explicit DeviceCtx(const PhiDeviceCtx* phi_device_ctx)
      : phi_device_ctx_(phi_device_ctx) {}

  adt::Result<axpr::PointerValue> GetStreamAddrAsVoidPtr() override {
    if (!stream_.has_value()) {
      stream_ = reinterpret_cast<void*>(phi_device_ctx_->stream());
    }
    void* stream_ptr = std::any_cast<void*>(&stream_);
    return axpr::PointerValue{stream_ptr};
  }
};

}  // namespace ap::paddle
