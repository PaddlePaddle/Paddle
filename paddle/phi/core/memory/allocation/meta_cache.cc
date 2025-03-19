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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/memory/allocation/memory_block.h"

namespace paddle::memory::detail {

MetadataCache::MetadataCache(bool uses_gpu) : uses_gpu_(uses_gpu) {}

MemoryBlock::Desc* MetadataCache::LoadDesc(MemoryBlock* block) {
  if (uses_gpu_) {
    auto iter = cache_.find(block);
    PADDLE_ENFORCE_NE(
        iter,
        cache_.end(),
        common::errors::NotFound("The memory block is not found in cache"));
    auto* desc = &(iter->second);
    PADDLE_ENFORCE_EQ(
        desc->CheckGuards(),
        true,
        common::errors::InvalidArgument("Invalid CPU memory access"));
    return desc;
  } else {
    auto* desc = reinterpret_cast<MemoryBlock::Desc*>(block);
    VLOG(10) << "Load MemoryBlock::Desc type=" << desc->type;
    PADDLE_ENFORCE_EQ(
        desc->CheckGuards(),
        true,
        common::errors::InvalidArgument("Invalid CPU memory access"));
    return reinterpret_cast<MemoryBlock::Desc*>(block);
  }
}

void MetadataCache::Save(MemoryBlock* block,
                         const MemoryBlock::Desc& original_desc) {
  auto desc = original_desc;
  desc.UpdateGuards();

  if (uses_gpu_) {
    cache_[block] = desc;
  } else {
    *reinterpret_cast<MemoryBlock::Desc*>(block) = desc;
  }
}

void MetadataCache::Invalidate(MemoryBlock* block) {
  if (uses_gpu_) {
    cache_.erase(block);
  }
}

}  // namespace paddle::memory::detail
