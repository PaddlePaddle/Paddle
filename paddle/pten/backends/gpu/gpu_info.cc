/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/backends/gpu/gpu_info.h"

#include <vector>

#include "gflags/gflags.h"

DECLARE_string(selected_gpus);

namespace pten {
namespace backends {
namespace gpu {

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
std::vector<int> GetSelectedDevices() {
  // use user specified GPUs in single-node multi-process mode.
  std::vector<int> devices;
  if (!FLAGS_selected_gpus.empty()) {
    auto devices_str = Split(FLAGS_selected_gpus, ',');
    for (auto id : devices_str) {
      devices.push_back(atoi(id.c_str()));
    }
  } else {
    int count = GetGPUDeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }
  return devices;
}

}  // namespace gpu
}  // namespace backends
}  // namespace pten
