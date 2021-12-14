// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <glog/logging.h>
#include <unordered_map>

#include <memory>

#include "paddle/infrt/common/macros.h"
#include "paddle/infrt/common/target.h"

namespace infrt {

class MemoryInterface {
 public:
  virtual void* malloc(size_t nbytes) = 0;
  virtual void free(void* data) = 0;
  virtual void* aligned_alloc(size_t alignment, size_t nbytes) {
    return nullptr;
  }
  virtual ~MemoryInterface() {}
};

/**
 * MemoryManager holds a map of MemoryInterface for each articture.
 */
class MemoryManager final {
 public:
  using key_t = common::Target::Arch;

  static MemoryManager& Global() {
    static auto* x = new MemoryManager;
    return *x;
  }

  MemoryInterface* Retrieve(key_t key) INFRT_RESULT_SHOULD_USE {
    auto it = memory_mngs_.find(key);
    if (it != memory_mngs_.end()) return it->second.get();
    return nullptr;
  }

  MemoryInterface* RetrieveSafely(key_t key) {
    auto* res = Retrieve(key);
    CHECK(res) << "no MemoryInterface for architecture [" << key << "]";
    return res;
  }

  MemoryInterface* Register(key_t key, MemoryInterface* item) {
    CHECK(!memory_mngs_.count(key)) << "Duplicate register [" << key << "]";
    memory_mngs_[key].reset(item);
    return item;
  }

 private:
  MemoryManager();

  std::unordered_map<common::Target::Arch, std::unique_ptr<MemoryInterface>>
      memory_mngs_;

  INFRT_DISALLOW_COPY_AND_ASSIGN(MemoryManager);
};

}  // namespace infrt
