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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/enforce.h"

#if defined(PADDLE_WITH_CUSTOM_DEVICE)
#include "paddle/phi/backends/device_manager.h"
#define CONTEXT_TYPE CustomContext
#elif defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#define CONTEXT_TYPE GPUContext
#else
#define CONTEXT_TYPE DeviceContext
#endif

namespace phi {

namespace funcs {
inline CONTEXT_TYPE *GetCurrentContext() {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  int device_id = phi::DeviceManager::GetDevice(dev_types[0]);
  auto gplace = phi::CustomPlace(dev_types[0], device_id);
  auto *dev_ctx = static_cast<CustomContext *>(
      phi::DeviceContextPool::Instance().Get(gplace));
  return dev_ctx;
#elif defined(__NVCC__) || defined(__HIPCC__)
  auto gplace = phi::GPUPlace(phi::backends::gpu::GetCurrentDeviceId());
  auto *dev_ctx =
      static_cast<GPUContext *>(phi::DeviceContextPool::Instance().Get(gplace));
  return dev_ctx;
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "Unsupported usage of phi::funcs::GetCurrentContext()! This function "
      "only support CUDA and Custom Device."));
  return 0;
#endif
}
}  // namespace funcs

}  // namespace phi
