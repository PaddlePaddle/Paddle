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
#include <memory>
#include <string>
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {
namespace allocation {

// Exception when `Alloc`/`AllocShared` failed
class BadAlloc : public std::exception {
 public:
  explicit BadAlloc(std::string msg) : msg_(std::move(msg)) {}
  const char* what() const noexcept override;

 private:
  std::string msg_;
};

// Allocation is the object holding the actually pointer. Use
// `Allocation::ptr()` will returns the pointer that allocated.
//
// NOTE: this is the base class of Allocation. Each allocator can use its own
//       allocation object.
// NOTE: the `Allocation::ptr()` could be nullptr, if the allocation size is 0
class Allocation {
 public:
  Allocation(void* ptr, size_t size, platform::Place place)
      : ptr_(ptr), size_(size), place_(place) {}

  Allocation(const Allocation& o) = delete;
  Allocation& operator=(const Allocation& o) = delete;

  // Returns the holding pointer.
  // NOTE: For performance consideration, it is better not to make this method
  // as a virtual method. If we want to implement a `defragmentation` later,
  // we might need to make `ptr_` field as a protected field, and add a virtual
  // method like `defragmentation` to change `ptr_`.
  void* ptr() const { return ptr_; }

  // Returns the size of this memory buffer, i.e., ptr() + size() - 1 is the
  // last valid element.
  //
  // NOTE: Some allocator might alloc more memory than request. The size
  // could larger than its request. For example,
  //    the AlignedAllocator will always allocate memory as size + kAlignment.
  //    The raw pointer might not aligned, so an offset might be added to raw
  //    the pointer. The size of this allocation will be
  //    `size + kAlignemnt - offset`.
  size_t size() const { return size_; }

  const platform::Place& place() const { return place_; }

  virtual ~Allocation();

 private:
  void* ptr_;
  size_t size_;
  platform::Place place_;
};

// Base interface class of memory Allocator.
// To allocate a memory, allocator needs two parameters:
//    1. size of bytes.
//    2. Attribute of memory.
// NOTE: the attribute of memory might be ignored if the allocator does not
// care it.
class Allocator {
 public:
  enum Attr {
    kDefault = 0,  // Default attribute. Uses the fast or stablest allocation
                   // algorithm.

    kFixedHuge = 1,  // The allocation may not be freed until the program
                     // ends. e.g., `Parameters` and `Momentum`.

    kFluxHuge = 2,  // The allocation may create and freed frequently and the
                    // allocation is considerable huge. Like `activations`
                    // and gradients.

    kScratchpad =
        3,  // The `Scratchpad` memory is allocated and freed very soon,
            // usually within an operator or aux memory.
            // Like CUDNN workspace, AUX memory in batch norm, etc.
            //
            // https://en.wikipedia.org/wiki/Scratchpad_memory

    kCrossDevice =
        4,  // The memory used cross-device memory copy/communication.
            // For example:
            // 1. it can use an `pinned` memory for CPU-GPU
            //    communication.
            // 2. it can use an `registered` memory for RDMA
            //    communication.

    NumOfAttrs = 5  // The number of all attributes. It is used internally.
  };

  virtual ~Allocator();

  // Allocate an allocation. Note the return allocation might need to be freed
  // manually if the Allocator is an `UnmanagedAllocator`.
  virtual std::unique_ptr<Allocation> Allocate(
      size_t size, Allocator::Attr attr = kDefault) = 0;

  // True if the `Allocate` is thread safe.
  virtual bool IsAllocThreadSafe() const;
};

class MannualFreeAllocator;
class MannualFreeAllocation : public Allocation {
 public:
  MannualFreeAllocation(MannualFreeAllocator* allocator, void* ptr, size_t size,
                        platform::Place place)
      : Allocation(ptr, size, place), allocator_(allocator) {}

  ~MannualFreeAllocation();

 private:
  MannualFreeAllocator* allocator_;
};

// User need to invoke `Free` or `FreeUniquePtr` manually if allocated by
// a manally managed allocator.
class MannualFreeAllocator : public Allocator {
 public:
  std::unique_ptr<Allocation> Allocate(size_t size, Attr attr) final;

 protected:
  virtual void Free(MannualFreeAllocation* allocation) = 0;
  virtual MannualFreeAllocation* AllocateImpl(size_t size,
                                              Allocator::Attr attr) = 0;
  friend class MannualFreeAllocation;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
