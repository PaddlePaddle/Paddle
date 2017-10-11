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

#include "Storage.h"
#include "Allocator.h"
#include "paddle/utils/StringUtil.h"
#include "paddle/utils/Util.h"

DEFINE_int32(pool_limit_size,
             536870912,
             "maximum memory size managed by a memory pool, default is 512M");

namespace paddle {

// Initialization StorageEngine singleton.
// Other modules may rely on storage management,
// so StorageEngine need to be initialized before other modules.
static InitFunction __init_storage_engine([]() { StorageEngine::singleton(); },
                                          std::numeric_limits<int>::max());

StorageEngine::StorageEngine() : cpuAllocator_(nullptr) {}

StorageEngine::~StorageEngine() {
  if (cpuAllocator_) {
    delete cpuAllocator_;
  }
  for (auto it : gpuAllocator_) {
    delete it;
  }
}

StorageEngine* StorageEngine::singleton() {
  static StorageEngine storage;
  return &storage;
}

PoolAllocator* StorageEngine::getGpuAllocator(int deviceId) {
  {
    // if gpuAllocator_ has been constructed
    ReadLockGuard guard(lock_);
    if (deviceId < static_cast<int>(gpuAllocator_.size()) &&
        (gpuAllocator_[deviceId] != nullptr)) {
      return gpuAllocator_[deviceId];
    }
  }

  {
    // Construct gpuAllocator_
    std::lock_guard<RWLock> guard(lock_);
    if (deviceId >= static_cast<int>(gpuAllocator_.size())) {
      gpuAllocator_.resize(deviceId + 1);
    }
    if (gpuAllocator_[deviceId] == nullptr) {
      std::string name =
          "gpu" + str::to_string(deviceId) + std::string("_pool");
      gpuAllocator_[deviceId] =
          new PoolAllocator(new GpuAllocator(), FLAGS_pool_limit_size, name);
    }
    return gpuAllocator_[deviceId];
  }
}

PoolAllocator* StorageEngine::getCpuAllocator() {
  {
    // if cpuAllocator_ has been constructed
    ReadLockGuard guard(lock_);
    if (cpuAllocator_ != nullptr) {
      return cpuAllocator_;
    }
  }

  {
    // Construct cpuAllocator_
    std::lock_guard<RWLock> guard(lock_);
    if (cpuAllocator_ == nullptr) {
      if (FLAGS_use_gpu) {
        cpuAllocator_ = new PoolAllocator(
            new CudaHostAllocator(), FLAGS_pool_limit_size, "cuda_host_pool");
      } else {
        cpuAllocator_ = new PoolAllocator(
            new CpuAllocator(), FLAGS_pool_limit_size, "cpu_pool");
      }
    }
    return cpuAllocator_;
  }
}

}  // namespace paddle
