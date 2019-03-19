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
#pragma once

#include <cstdint>
#include <unordered_map>

namespace paddle {
namespace memory {
namespace detail {

// Forward declaration.
class MetadataCache;

// MemoryBlock represents Each allocated memory block, which contains
// MemoryBlock::Desc and the payload.
struct MemoryBlock {
  enum Type {
    FREE_CHUNK,    // memory is free and idle
    ARENA_CHUNK,   // memory is being occupied
    HUGE_CHUNK,    // memory is out of management
    INVALID_CHUNK  // memory is invalid
  };

  // init saves the MemoryBlock::Desc of the memory block in a MetadataCache.
  // If it is a CPU memory block, the MetadataCache writes the
  // MemoryBlock::Desc to the beginning of the block; or, if it is a GPU memory
  // block, the MetadataCache writes the Meatadata to a std::map in
  // the CPU.
  void init(MetadataCache* cache, Type t, size_t index, size_t size,
            void* left_buddy, void* right_buddy);

  // All these accessors returns fields in the MemoryBlock::Desc of the memory
  // block.  They all need a MetadataCache instance as their first
  // parameter because they read the MemoryBlock::Desc from the cache.
  Type type(const MetadataCache& cache) const;
  size_t size(const MetadataCache& cache) const;
  size_t index(const MetadataCache& cache) const;
  size_t total_size(const MetadataCache& cache) const;
  bool has_left_buddy(const MetadataCache& cache) const;
  bool has_right_buddy(const MetadataCache& cache) const;
  MemoryBlock* left_buddy(const MetadataCache& cache) const;
  MemoryBlock* right_buddy(const MetadataCache& cache) const;

  // Split the allocation into left/right blocks.
  void split(MetadataCache* cache, size_t size);

  // Merge left and right blocks together.
  void merge(MetadataCache* cache, MemoryBlock* right_buddy);

  // Mark the allocation as free.
  void mark_as_free(MetadataCache* cache);

  // Change the type of the allocation.
  void set_type(MetadataCache* cache, Type t);

  void* data() const;
  MemoryBlock* metadata() const;

  // MemoryBlock::Desc describes a MemoryBlock.
  struct Desc {
    Desc(MemoryBlock::Type t, size_t i, size_t s, size_t ts, MemoryBlock* l,
         MemoryBlock* r);
    Desc();

    // Updates guard_begin and guard_end by hashes of the Metadata object.
    void update_guards();

    // Checks that guard_begin and guard_end are hashes of the Metadata object.
    bool check_guards() const;

    // TODO(gangliao): compress this
    size_t guard_begin = 0;
    MemoryBlock::Type type = MemoryBlock::INVALID_CHUNK;
    size_t index = 0;
    size_t size = 0;
    size_t total_size = 0;
    MemoryBlock* left_buddy = nullptr;
    MemoryBlock* right_buddy = nullptr;
    size_t guard_end = 0;
  };
};

// A cache for accessing memory block meta-data that may be expensive
// to access directly.  This class exists to unify the
// MemoryBlock::Desc format between GPU and CPU allocations. It should
// be removed when the CPU can access all GPU allocations directly via
// UVM.
class MetadataCache {
 public:
  explicit MetadataCache(bool uses_gpu);

  // Disable copying and assignment.
  MetadataCache(const MetadataCache&) = delete;
  MetadataCache& operator=(const MetadataCache&) = delete;

  // Returns the MemoryBlock::Desc for a memory block.  When MetadataCache is
  // used to manage CPU memory, the MemoryBlock::Desc resides at the beginning
  // of the memory block; when used to manage GPU memory, the
  // Meatadata resides in CPU memory indexed by cache_.
  MemoryBlock::Desc load(const MemoryBlock* memory_block) const;

  // Saves the MemoryBlock::Desc of a memory block into the cache.  For CPU
  // memory block, writes the MemoryBlock::Desc to the beginning of the memory
  // block; whereas for GPU memory, writes it to cache_.
  void save(MemoryBlock* memory_block, const MemoryBlock::Desc& meta_data);

  // For GPU memory block, erases its MemoryBlock::Desc from cache_.
  void invalidate(MemoryBlock* memory_block);

 private:
  typedef std::unordered_map<const MemoryBlock*, MemoryBlock::Desc> MetadataMap;
  MetadataMap cache_;
  bool uses_gpu_;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
