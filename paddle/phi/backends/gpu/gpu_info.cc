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

#include "paddle/phi/backends/gpu/gpu_info.h"

#include <sstream>
#include <vector>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/common/memory_utils.h"

COMMON_DECLARE_string(selected_gpus);

namespace phi::backends::gpu {

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
    for (auto const& id : devices_str) {
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

constexpr static float fraction_reserve_gpu_memory = 0.05f;

size_t GpuAvailableMemToAlloc() {
  size_t total = 0;
  size_t available = 0;
  memory_utils::GpuMemoryUsage(&available, &total);
  size_t reserving =
      static_cast<size_t>(fraction_reserve_gpu_memory * available);  // NOLINT
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = available - reserving;
  size_t min_chunk_size = GpuMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  VLOG(10) << "GPU usage " << (available >> 20) << "M/" << (total >> 20)
           << "M, " << (available_to_alloc >> 20) << "M available to allocate";
  return available_to_alloc;
}

size_t GpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 256 bytes.
  return 1 << 8;
}

}  // namespace phi::backends::gpu
