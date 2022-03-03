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
#include "paddle/phi/backends/xpu/xpu_info.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/common/place.h"

// TODO(wilber): The phi computing library requires a component to manage
// flags.
#include "paddle/fluid/platform/flags.h"

PADDLE_DEFINE_EXPORTED_string(
    selected_xpus,
    "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (XPU). If you want to use "
    "all visible devices, set this to empty string. NOTE: the "
    "reason of doing this is that we want to use P2P communication"
    "between XPU devices, use XPU_VISIBLE_DEVICES can only use"
    "share-memory only.");

namespace phi {
class XPUContext;

namespace backends {
namespace xpu {

/**************************** Version Management **************************/

//! Get the version of XPU Driver
int GetDriverVersion() {
  uint32_t driver_version_major = 0;
  uint32_t driver_version_minor = 0;
  PADDLE_ENFORCE_XPU_SUCCESS(
      xpu_get_driver_version(&driver_version_major, &driver_version_minor));
  int driver_version = driver_version_major * 10 + driver_version_minor;
  return driver_version;
}

//! Get the version of XPU Runtime
int GetRuntimeVersion() {
  uint32_t rumtime_version_major = 0;
  uint32_t rumtime_version_minor = 0;
  PADDLE_ENFORCE_XPU_SUCCESS(
      xpu_get_runtime_version(&rumtime_version_major, &rumtime_version_minor));
  int runtime_version = rumtime_version_major * 10 + rumtime_version_minor;
  return runtime_version;
}

/**************************** Device Management **************************/

static int GetDeviceCountImpl() {
  const auto* xpu_visible_devices = std::getenv("XPU_VISIBLE_DEVICES");
  if (xpu_visible_devices != nullptr) {
    std::string xpu_visible_devices_str(xpu_visible_devices);
    if (std::all_of(xpu_visible_devices_str.begin(),
                    xpu_visible_devices_str.end(),
                    [](char ch) { return ch == ' '; })) {
      VLOG(2) << "XPU_VISIBLE_DEVICES is set to be empty. No XPU detected.";
      return 0;
    }
  }

  int count = 0;
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_device_count(&count));
  return count;
}

int GetXPUDeviceCount() {
  static auto dev_cnt = GetDeviceCountImpl();
  return dev_cnt;
}

int GetXPUCurrentDeviceId() {
  int dev_id;
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_current_device(&dev_id));
  if (dev_id >= 64) {
    // if dev_id >= 64, the device is a simulator device, -64 to get real dev_id
    dev_id -= 64;
  }
  return dev_id;
}

void SetXPUDeviceId(int id) {
  PADDLE_ENFORCE_LT(
      id,
      GetXPUDeviceCount(),
      phi::errors::InvalidArgument("id must less than XPU count"));
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_set_device(id));
}

static inline std::vector<std::string> Split(std::string const& original,
                                             char separator) {
  std::vector<std::string> results;
  std::string token;
  std::istringstream is(original);
  while (std::getline(is, token, separator)) {
    if (!token.empty()) {
      results.push_back(token);
    }
  }
  return results;
}

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices() {
  // use user specified XPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_xpus.empty()) {
    auto devices_str = Split(FLAGS_selected_xpus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetXPUDeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

/**************************** Memory Management **************************/

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const phi::XPUPlace& dst_place) {
  XPUDeviceGuard guard(dst_place.device);
  PADDLE_ENFORCE_XPU_SUCCESS(
      xpu_memcpy(dst, src, count, XPUMemcpyKind::XPU_HOST_TO_DEVICE));
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const phi::XPUPlace& src_place,
                   const phi::XPUContext& dev_ctx) {
  XPUDeviceGuard guard(src_place.GetDeviceId());
  dev_ctx.Wait();
  PADDLE_ENFORCE_XPU_SUCCESS(
      xpu_memcpy(dst, src, count, XPUMemcpyKind::XPU_DEVICE_TO_HOST));
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst,
                   const phi::XPUPlace& dst_place,
                   const void* src,
                   const phi::XPUPlace& src_place,
                   size_t count,
                   const phi::XPUContext& dev_ctx) {
  int dev_id = GetXPUCurrentDeviceId();
  if (dst_place.device == dev_id && src_place.device == dev_id) {
    PADDLE_ENFORCE_XDNN_SUCCESS(
        baidu::xpu::api::copy(dev_ctx.x_context(),
                              static_cast<const int8_t*>(src),
                              static_cast<int8_t*>(dst),
                              count),
        "copy ");
  } else {
    PADDLE_ENFORCE_XPU_SUCCESS(
        xpu_memcpy_peer(dst_place.device, dst, src_place.device, src, count));
  }
}

/**************************** Others **************************/

XPUVersion get_xpu_version(int dev_id) {
  uint64_t v = 0;
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_device_get_attr(&v, XPUATTR_MODEL, dev_id));

  if (v == K100 || v == K200) {
    VLOG(1) << "KUNLUN device " << dev_id << " is XPU1\n";
    return XPU1;
  } else {
    VLOG(1) << "KUNLUN device " << dev_id << " is XPU2\n";
    return XPU2;
  }
}

}  // namespace xpu
}  // namespace backends
}  // namespace phi
