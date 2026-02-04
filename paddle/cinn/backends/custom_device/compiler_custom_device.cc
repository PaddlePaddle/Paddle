// Copyright (c) 2026 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/custom_device/compiler_custom_device.h"
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"

#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <glog/logging.h>
#include <fstream>
#include <iostream>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/runtime/custom_device/custom_device_util.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/place.h"

namespace cinn {
namespace backends {
namespace cdrtc {

Compiler::Compiler(const cinn::common::Target& target) : target_(target) {}
std::string Compiler::operator()(const std::string& code,
                                 bool include_headers) {
  std::string dev_type = "";
  auto devs = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (!devs.empty()) {
    // Default to the first registered custom device
    // Notice: Multi-vendor Environment not supported yet
    dev_type = devs[0];
  }

  auto place = phi::CustomPlace(dev_type, 0);
  // 1. Get the plugin instance
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);

  // 2. Forward the compilation request to the plugin's Toolchain
  return plugin.GetToolchain()->Compile(code);
}

}  // namespace cdrtc
}  // namespace backends
}  // namespace cinn
