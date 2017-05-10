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

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "Allocator.h"

namespace paddle {

/**
 * @brief Memory pool allocator implementation.
 */
class PoolAllocator {
public:
  /**
   * @brief constructor.
   * @param allocator a Allocator object.
   * @param sizeLimit The maximum size memory can be managed,
   * if sizeLimit == 0, the pool allocator is a simple wrapper of allocator.
   */
  PoolAllocator(Allocator* allocator,
                size_t sizeLimit = 0,
                const std::string& name = "pool");

  /**
   * @brief destructor.
   */
  ~PoolAllocator();

  void* alloc(size_t size);
  void free(void* ptr, size_t size);
  std::string getName() { return name_; }

private:
  void freeAll();
  void printAll();
  std::unique_ptr<Allocator> allocator_;
  std::mutex mutex_;
  std::unordered_map<size_t, std::vector<void*>> pool_;
  size_t sizeLimit_;
  size_t poolMemorySize_;
  std::string name_;
};

}  // namespace paddle
