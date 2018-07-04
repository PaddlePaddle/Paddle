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

#include "glog/logging.h"
#include "paddle/fluid/memory/detail/memory_block.h"
#include "paddle/fluid/platform/assert.h"

namespace paddle {
namespace memory {
namespace detail {

MetadataCache::MetadataCache(bool uses_gpu) : uses_gpu_(uses_gpu) {}

MemoryBlock::Desc MetadataCache::load(const MemoryBlock* block) const {
  if (uses_gpu_) {
    auto existing_desc = cache_.find(block);
    PADDLE_ASSERT(existing_desc->second.check_guards());
    return existing_desc->second;
  } else {
    auto* desc = reinterpret_cast<const MemoryBlock::Desc*>(block);
    VLOG(10) << "Load MemoryBlock::Desc type=" << desc->type;
    PADDLE_ASSERT(desc->check_guards());
    return *reinterpret_cast<const MemoryBlock::Desc*>(block);
  }
}

void MetadataCache::save(MemoryBlock* block,
                         const MemoryBlock::Desc& original_desc) {
  auto desc = original_desc;
  desc.update_guards();

  if (uses_gpu_) {
    cache_[block] = desc;
  } else {
    *reinterpret_cast<MemoryBlock::Desc*>(block) = desc;
  }
}

void MetadataCache::invalidate(MemoryBlock* block) {
  if (uses_gpu_) {
    cache_.erase(block);
  }
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
