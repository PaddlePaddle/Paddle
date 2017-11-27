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

#include "paddle/memory/detail/meta_cache.h"
#include "glog/logging.h"
#include "paddle/memory/detail/memory_block.h"
#include "paddle/platform/assert.h"

namespace paddle {
namespace memory {
namespace detail {

MetadataCache::MetadataCache(bool uses_gpu) : uses_gpu_(uses_gpu) {}

Metadata MetadataCache::load(const MemoryBlock* block) {
  if (uses_gpu_) {
    auto existing_metadata = cache_.find(block);
    PADDLE_ASSERT(existing_metadata->second.check_guards());
    return existing_metadata->second;
  } else {
    auto* meta = reinterpret_cast<const Metadata*>(block);
    VLOG(10) << "Load MetaData type=" << meta->type;
    PADDLE_ASSERT(meta->check_guards());
    return *reinterpret_cast<const Metadata*>(block);
  }
}

void MetadataCache::store(MemoryBlock* block,
                          const Metadata& original_metadata) {
  auto metadata = original_metadata;

  metadata.update_guards();

  if (uses_gpu_) {
    cache_[block] = metadata;
  } else {
    *reinterpret_cast<Metadata*>(block) = metadata;
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
