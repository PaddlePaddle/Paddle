// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdint.h>

#include <array>
#include <list>
#include <map>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/memory/allocation/allocator.h"
#include "paddle/phi/core/memory/allocation/spin_lock.h"

namespace paddle {
namespace memory {
namespace allocation {
namespace details {
struct Chunk {
  bool is_free{true};
  // Offset to the base allocation.
  uintptr_t offset_;
  size_t size_;
};

// Here we use std::list to maintain chunk list.
// NOTE(yy): The traditional implementation of ChunkList is add `prev`/`next`
// pointers in `Chunk`, and split the allocation as `ChunkHeader` and
// `Payload`. Such as
//   *-------*---------------*---------------*--------------*
//   | Chunk | prev_ pointer | next_ pointer | payload .... |
//   *-------*---------------*---------------*--------------*
// This implementation can just return a raw pointer, and we can get the list
// structure by the raw pointer. However, we cannot use the same code on GPU
// since CPU cannot access GPU memory directly.
//
// So we choose to use `std::list` and return an allocation instance, which
// contains the list node iterator, then we can unify CPU/GPU code.
//
// To return an allocation is not a bad idea, since Tensor/Vector should holds
// an allocation instead of raw pointer directly.
using ChunkList = std::list<Chunk>;

// Here we use a multi-level map of free chunks.
// the map is
//      MSB offset --> size --> [ChunkList::iterator]
//
// The time complexities:
//     find a free chunk:
//          O(logN),
//               where N is the number of free nodes with the same MSB offset.
//     find the position of a chunk iterator:
//          O(logN + K),
//               where N is the number of free nodes with the same MSB offset.
//               where K is the number of free nodes with the same size.
//     insert a free chunk:
//          O(logN),
//               where N is the number of free nodes with the same MSB offset.
//     erase a free chunk:
//          O(1)
using FreeChunkBin =
    std::array<std::multimap<size_t, ChunkList::iterator>, sizeof(size_t) * 8>;
}  // namespace details

class BestFitAllocator;

// The BestFitAllocation maintain the List Node iterator.
class BestFitAllocation : public Allocation {
 private:
  using ListIt = typename details::ChunkList::iterator;

 public:
  BestFitAllocation(BestFitAllocator* allocator, ListIt chunk_it);

  const ListIt& ChunkIterator() const { return chunk_it_; }

 private:
  typename details::ChunkList::iterator chunk_it_;
};

// TODO(yy): Current BestFitAllocator is not thread-safe. To make it thread
// safe, we must wrap a locked_allocator. However, we can implement a thread
// safe allocator by locking each bin and chunks list independently. It will
// make BestFitAllocator faster in multi-thread situation.
//
// This allocator implements a best-fit allocator with merging the free nodes.
//
// To allocate a buffer, it will find the best-fit chunk. If the best-fit chunk
// is larger than request size, the original block will be split into two
// chunks. The first block will be used and the second block will be put into
// free chunks.
//
// To free an allocation, it will set the chunk of allocation to free and merge
// the prev-chunk and the next-chunk when possible.
class BestFitAllocator : public Allocator {
 public:
  explicit BestFitAllocator(phi::Allocation* allocation);

  void* BasePtr() const { return allocation_->ptr(); }

  const phi::Place& Place() const { return allocation_->place(); }

  size_t NumFreeChunks() const;

  bool IsAllocThreadSafe() const override { return true; }

 private:
  size_t FreeSize() const;
  using MapIt = typename details::FreeChunkBin::value_type::iterator;
  using ListIt = typename details::ChunkList::iterator;

  ListIt SplitChunk(size_t request_size,
                    size_t free_chunk_offset,
                    MapIt bin_iterator);
  void EraseFreeNode(const ListIt& it);
  void InsertFreeNode(const ListIt& it);

 protected:
  void FreeImpl(phi::Allocation* allocation) override;
  phi::Allocation* AllocateImpl(size_t size) override;

 private:
  phi::Allocation* allocation_;  // not owned
  details::ChunkList chunks_;
  details::FreeChunkBin free_chunks_;
  SpinLock spinlock_;
};
}  // namespace allocation
}  // namespace memory
}  // namespace paddle
