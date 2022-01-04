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
#pragma once

#ifdef PADDLE_WITH_XPU
#include <vector>

namespace paddle {
namespace platform {

/***** Version Management *****/

//! Get the version of XPU Driver
int GetDriverVersion();

//! Get the version of XPU Runtime
int GetRuntimeVersion();

/***** Device Management *****/

//! Get the total number of XPU devices in system.
int GetXPUDeviceCount();

//! Set the XPU device id for next execution.
void SetXPUDeviceId(int device_id);

//! Get the current XPU device id in system.
int GetXPUCurrentDeviceId();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices();

/***** Memory Management *****/

//! Copy memory from address src to dst synchronously.
void MemcpySyncH2D(void *dst, const void *src, size_t count, int dev_id);
void MemcpySyncD2H(void *dst, const void *src, size_t count, int dev_id);
void MemcpySyncD2D(void *dst, int dst_id, const void *src, int src_id,
                   size_t count);

class XPUDeviceGuard {
 public:
  explicit inline XPUDeviceGuard(int dev_id) {
    int prev_id = platform::GetXPUCurrentDeviceId();
    if (prev_id != dev_id) {
      prev_id_ = prev_id;
      platform::SetXPUDeviceId(dev_id);
    }
  }

  inline ~XPUDeviceGuard() {
    if (prev_id_ != -1) {
      platform::SetXPUDeviceId(prev_id_);
    }
  }

  XPUDeviceGuard(const XPUDeviceGuard &o) = delete;
  XPUDeviceGuard &operator=(const XPUDeviceGuard &o) = delete;

 private:
  int prev_id_{-1};
};

enum XPUVersion { XPU1, XPU2 };
XPUVersion get_xpu_version(int dev_id);

}  // namespace platform
}  // namespace paddle
#endif
