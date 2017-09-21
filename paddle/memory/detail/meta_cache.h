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

#include "paddle/memory/detail/memory_block.h"
#include "paddle/memory/detail/meta_data.h"

#include <unordered_map>

namespace paddle {
namespace memory {
namespace detail {

/**
 *  \brief A cache for accessing memory block meta-data that may be expensive
 *         to access directly.
 *
 *  \note  This class exists to unify the metadata format between GPU and CPU
 *         allocations. It should be removed when the CPU can access all GPU
 *         allocations directly via UVM.
 */
class MetadataCache {
 public:
  explicit MetadataCache(bool uses_gpu);

 public:
  /*! \brief Load the associated metadata for the specified memory block. */
  Metadata load(const MemoryBlock* memory_block);

  /*! \brief Store the associated metadata for the specified memory block. */
  void store(MemoryBlock* memory_block, const Metadata& meta_data);

  /*! \brief Indicate that the specified metadata will no longer be used. */
  void invalidate(MemoryBlock* memory_block);

 public:
  MetadataCache(const MetadataCache&) = delete;
  MetadataCache& operator=(const MetadataCache&) = delete;

 private:
  bool uses_gpu_;

 private:
  typedef std::unordered_map<const MemoryBlock*, Metadata> MetadataMap;

 private:
  MetadataMap cache_;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
