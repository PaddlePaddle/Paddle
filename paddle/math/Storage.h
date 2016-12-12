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

#pragma once

#include <mutex>
#include <vector>
#include "PoolAllocator.h"
#include "paddle/utils/Locks.h"

namespace paddle {

/**
 * @brief Storage manager for multiple devices.
 */
class StorageEngine {
public:
  /**
   * @return Storage singleton
   */
  static StorageEngine* singleton();

  /**
   * @return return one gpu allocator by deviceId
   */
  PoolAllocator* getGpuAllocator(int deviceId);

  /**
   * @return return cpu allocator
   */
  PoolAllocator* getCpuAllocator();

protected:
  StorageEngine();
  ~StorageEngine();
  RWLock lock_;
  std::vector<PoolAllocator*> gpuAllocator_;
  PoolAllocator* cpuAllocator_;
};

}  // namespace paddle
