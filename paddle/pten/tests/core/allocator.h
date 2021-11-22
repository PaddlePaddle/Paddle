/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>

#include "paddle/pten/core/allocator.h"

namespace pten {
namespace tests {

class HostAllocatorSample : public pten::RawAllocator {
 public:
  using Place = paddle::experimental::Place;
  using DeviceType = paddle::experimental::DeviceType;

  void* Allocate(size_t bytes_size) override {
    return ::operator new(bytes_size);
  }
  void Deallocate(void* ptr, size_t bytes_size) override {
    return ::operator delete(ptr);
  }
  const Place& place() const override { return place_; }

 private:
  Place place_{paddle::experimental::DeviceType::kHost, 0};
};

class FancyAllocator : public pten::Allocator {
 public:
  using Place = paddle::experimental::Place;
  using DeviceType = paddle::experimental::DeviceType;

  static void Delete(void* data) { ::operator delete(data); }

  Allocation Allocate(size_t bytes_size) override {
    void* data = ::operator new(bytes_size);
    return Allocation(data, data, &Delete, Place(DeviceType::kHost, 0));
  }
};

template <typename T>
struct CustomAllocator {
  using value_type = T;
  using Allocator = pten::RawAllocator;

  explicit CustomAllocator(const std::shared_ptr<Allocator>& a) noexcept
      : alloc_(a) {}

  CustomAllocator(const CustomAllocator&) noexcept = default;
  T* allocate(std::size_t n) {
    return static_cast<T*>(alloc_->Allocate(n * sizeof(T)));
  }
  void deallocate(T* p, std::size_t n) {
    return alloc_->Deallocate(p, sizeof(T) * n);
  }

  template <typename R, typename U>
  friend bool operator==(const CustomAllocator<R>&,
                         const CustomAllocator<U>&) noexcept;
  template <typename R, typename U>
  friend bool operator!=(const CustomAllocator<R>&,
                         const CustomAllocator<U>&) noexcept;

 private:
  std::shared_ptr<Allocator> alloc_;
};

template <typename T, typename U>
inline bool operator==(const CustomAllocator<T>& lhs,
                       const CustomAllocator<U>& rhs) noexcept {
  return &lhs.alloc_ == &rhs.alloc_;
}

template <typename T, typename U>
inline bool operator!=(const CustomAllocator<T>& lhs,
                       const CustomAllocator<U>& rhs) noexcept {
  return &lhs.alloc_ != &rhs.alloc_;
}

}  // namespace tests
}  // namespace pten
