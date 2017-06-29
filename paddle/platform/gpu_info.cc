/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/gpu_info.h"
#include "gflags/gflags.h"
#include "paddle/platform/error.h"

DEFINE_double(fraction_of_gpu_memory_to_use, 0.95,
              "Default use 95% of GPU memory for PaddlePaddle,"
              "reserve the rest for page tables, etc");

namespace paddle {
namespace platform {

int GpuDeviceCount() {
  int count;
  throw_on_error(
      cudaGetDeviceCount(&count),
      "cudaGetDeviceCount failed in paddle::platform::GpuDeviceCount");
  return count;
}

void GpuMemoryUsage(size_t& available, size_t& total) {
  throw_on_error(cudaMemGetInfo(&available, &total),
                 "cudaMemGetInfo failed in paddle::platform::GetMemoryUsage");
}

size_t GpuMaxAllocSize() {
  size_t total = 0;
  size_t available = 0;

  GpuMemoryUsage(available, total);

  return total * FLAGS_fraction_of_gpu_memory_to_use;
}

}  // namespace platform
}  // namespace paddle
