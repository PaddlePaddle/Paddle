/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device/xpu/xpu_info.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include "gflags/gflags.h"

#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/pten/backends/xpu/xpu_info.h"

namespace paddle {
namespace platform {

/**************************** Version Management **************************/

//! Get the version of XPU Driver
int GetDriverVersion() { return pten::backends::xpu::GetDriverVersion(); }

//! Get the version of XPU Runtime
int GetRuntimeVersion() { return pten::backends::xpu::GetRuntimeVersion(); }

/**************************** Device Management **************************/

int GetXPUDeviceCount() { return pten::backends::xpu::GetXPUDeviceCount(); }

int GetXPUCurrentDeviceId() {
  return pten::backends::xpu::GetXPUCurrentDeviceId();
}

void SetXPUDeviceId(int id) { pten::backends::xpu::SetXPUDeviceId(id); }

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices() {
  // use user specified XPUs in single-node multi-process mode.
  return pten::backends::xpu::GetXPUSelectedDevices();
}

/**************************** Memory Management **************************/

void MemcpySyncH2D(void* dst, const void* src, size_t count,
                   const platform::XPUPlace& dst_place) {
  pten::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place);
}

void MemcpySyncD2H(void* dst, const void* src, size_t count,
                   const platform::XPUPlace& src_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  dev_ctx->Wait();
  pten::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst, const platform::XPUPlace& dst_place,
                   const void* src, const platform::XPUPlace& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  pten::backends::xpu::MemcpySyncD2D(dst, dst_place, src, src_place, count,
                                     *dev_ctx);
}

/**************************** Others **************************/

pten::backends::xpu::XPUVersion get_xpu_version(int dev_id) {
  return pten::backends::xpu::get_xpu_version(dev_id);
}

}  // namespace platform
}  // namespace paddle
