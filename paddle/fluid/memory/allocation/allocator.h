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

class BadAlloc : public std::exception {
 public:
  explicit BadAlloc(const std::string& msg) : msg_(msg) {}
  const char* what() const noexcept override;

 private:
  std::string msg_;
};

class Allocation {
 public:
  Allocation(void* ptr, size_t size, platform::Place place)
      : ptr_(ptr), size_(size), place_(place) {}

  Allocation(const Allocation& o) = delete;
  Allocation& operator=(const Allocation& o) = delete;

  void* ptr() const { return ptr_; }

  size_t size() const { return size_; }

  const platform::Place& place() const { return place_; }

  virtual ~Allocation();

 private:
  void* ptr_;
  size_t size_;
  platform::Place place_;
};

class Allocator {
 public:
  enum Attr {
    kDefault = 0,
    kTiny = 1,
    kFixedHuge = 2,
    kFluxHuge = 3,
    kTmp = 4,
    kCommunication = 5,
    NumOfAttrs = 6
  };

  virtual ~Allocator();
  virtual std::unique_ptr<Allocation> Allocate(
      size_t size, Allocator::Attr attr = kDefault) = 0;

  virtual bool IsAllocThreadSafe() const;
};

// User need to invoke `Free` or `FreeUniquePtr` manually if allocated by
// a manally managed allocator.
class UnmanagedAllocator : public Allocator {
 public:
  virtual void Free(Allocation* allocation) = 0;

  void FreeUniquePtr(std::unique_ptr<Allocation> allocation) {
    Free(allocation.get());
  }
};

// The allocation will be managed by smart pointers
class ManagedAllocator : public Allocator {
 public:
  virtual std::shared_ptr<Allocation> AllocateShared(
      size_t size, Allocator::Attr attr = kDefault) = 0;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
