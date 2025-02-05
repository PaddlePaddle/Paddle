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

#include <cstddef>
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
  // block, the MetadataCache writes the Metadata to a std::map in
  // the CPU.
  void Init(MetadataCache* cache,
            Type t,
            size_t index,
            size_t size,
            void* left_buddy,
            void* right_buddy);

  MemoryBlock* GetLeftBuddy(MetadataCache* cache);
  MemoryBlock* GetRightBuddy(MetadataCache* cache);

  // Split the allocation into left/right blocks.
  void Split(MetadataCache* cache, size_t size, size_t extra_padding_size = 0);

  // Merge left and right blocks together.
  void Merge(MetadataCache* cache, MemoryBlock* right_buddy);

  // Mark the allocation as free.
  void MarkAsFree(MetadataCache* cache);

  void* Data() const;
  MemoryBlock* Metadata() const;

  // MemoryBlock::Desc describes a MemoryBlock.
  struct Desc {
    Desc(MemoryBlock::Type t,
         size_t i,
         size_t s,
         size_t ts,
         MemoryBlock* l,
         MemoryBlock* r);
    Desc();

    // mutator for type
    inline void set_type(const MemoryBlock::Type& type) {
      this->type = type;
      this->UpdateGuards();
    }

    // accessor for type
    inline const MemoryBlock::Type& get_type() const { return this->type; }

    // accessor for index
    inline const size_t& get_index() const { return this->index; }

    // accessor for size
    inline const size_t& get_size() const { return this->size; }

    // accessor for total_size
    inline const size_t& get_total_size() const { return this->total_size; }

    // Updates guard_begin and guard_end by hashes of the Metadata object.
    void UpdateGuards();

    // Checks that guard_begin and guard_end are hashes of the Metadata object.
    bool CheckGuards() const;

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
  // Metadata resides in CPU memory indexed by cache_.
  MemoryBlock::Desc* LoadDesc(MemoryBlock* memory_block);

  // Saves the MemoryBlock::Desc of a memory block into the cache.  For CPU
  // memory block, writes the MemoryBlock::Desc to the beginning of the memory
  // block; whereas for GPU memory, writes it to cache_.
  void Save(MemoryBlock* memory_block, const MemoryBlock::Desc& meta_data);

  // For GPU memory block, erases its MemoryBlock::Desc from cache_.
  void Invalidate(MemoryBlock* memory_block);

 private:
  typedef std::unordered_map<const MemoryBlock*, MemoryBlock::Desc> MetadataMap;
  MetadataMap cache_;
  bool uses_gpu_;
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
