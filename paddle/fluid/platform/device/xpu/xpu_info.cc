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
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/split.h"

PADDLE_DEFINE_EXPORTED_string(
    selected_xpus, "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (XPU). If you want to use "
    "all visible devices, set this to empty string. NOTE: the "
    "reason of doing this is that we want to use P2P communication"
    "between XPU devices, use XPU_VISIBLE_DEVICES can only use"
    "share-memory only.");

namespace paddle {
namespace platform {

static int GetXPUDeviceCountImpl() {
  const auto *xpu_visible_devices = std::getenv("XPU_VISIBLE_DEVICES");
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
  int ret = xpu_device_count(&count);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
  return count;
}

int GetXPUDeviceCount() {
  static auto dev_cnt = GetXPUDeviceCountImpl();
  return dev_cnt;
}

int GetXPUCurrentDeviceId() {
  int dev_id;
  int ret = xpu_current_device(&dev_id);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));

  if (dev_id >= 64) {
    // if dev_id >= 64, the device is a simulator device, -64 to get real dev_id
    dev_id -= 64;
  }
  return dev_id;
}

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices() {
  // use user specified XPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_xpus.empty()) {
    auto devices_str = paddle::string::Split(FLAGS_selected_xpus, ',');
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

void SetXPUDeviceId(int id) {
  PADDLE_ENFORCE_LT(
      id, GetXPUDeviceCount(),
      platform::errors::InvalidArgument("id must less than XPU count"));
  int ret = xpu_set_device(id);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU API return wrong value[%d], please check whether "
                        "Baidu Kunlun Card is properly installed.",
                        ret));
}

XPUVersion get_xpu_version(int dev_id) {
  uint64_t v = 0;
  int ret = xpu_device_get_attr(&v, XPUATTR_MODEL, dev_id);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "xpu_device_get_attr return wrong value[%d]", ret));

  if (v == K100 || v == K200) {
    VLOG(1) << "KUNLUN device " << dev_id << " is XPU1\n";
    return XPU1;
  } else {
    VLOG(1) << "KUNLUN device " << dev_id << " is XPU2\n";
    return XPU2;
  }
}

}  // namespace platform
}  // namespace paddle
