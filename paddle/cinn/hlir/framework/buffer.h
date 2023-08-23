// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>

#include <memory>

#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/memory.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * Buffer helps to hold the memory, and offers a set of methods to help manage
 * the memory.
 */
struct Buffer final {
  Buffer() = default;
  explicit Buffer(const common::Target& target) { SetTarget(target); }
  ~Buffer() { Free(); }
  //! Resize the memory hold by this buffer *exactlly* to \p size.
  void Resize(uint32_t size);
  void Resize(uint32_t alignment, uint32_t size);

  //! Lazily resize the memory.
  void ResizeLazy(uint32_t size);
  void ResizeLazy(uint32_t alignment, uint32_t size);

  //! Resize the memory to \p size in target \p target.
  void Resize(uint32_t size, const common::Target& target);
  void Resize(uint32_t alignment, uint32_t size, const common::Target& target);

  //! Lazily resize the memory to \p size in target \p target.
  void ResizeLazy(uint32_t size, const common::Target& target);
  void ResizeLazy(uint32_t alignment,
                  uint32_t size,
                  const common::Target& target);

  void SetTarget(const common::Target& target);

  const cinn_buffer_t* data() const { return &data_; }
  cinn_buffer_t* data() { return &data_; }

  //! Free all the memory owned by this buffer.
  void Free() {
    if (!data_.memory) return;
    memory_mng_cache_->free(data_.memory);
  }

 private:
  inline void* Malloc(uint32_t size) CINN_RESULT_SHOULD_USE {
    CHECK(memory_mng_cache_) << "Should set target first";
    return memory_mng_cache_->malloc(size);
  }

  inline void* AlignedAlloc(uint32_t alignment,
                            uint32_t size) CINN_RESULT_SHOULD_USE {
    CHECK(memory_mng_cache_) << "Should set target first";
    return memory_mng_cache_->aligned_alloc(alignment, size);
  }

 private:
  cinn_buffer_t data_;

  //! The place where this buffer locates.
  common::Target target_;

  //! Number of bytes of this buffer.
  uint32_t size_{};

  //! Hold the corresponding memory manager for speed.
  MemoryInterface* memory_mng_cache_{};
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
