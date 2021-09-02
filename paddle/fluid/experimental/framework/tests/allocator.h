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

#include "paddle/fluid/experimental/framework/allocator.h"

namespace paddle {
namespace experimental {
namespace framework {
namespace tests {

class HostAllocatorSample : public experimental::framework::Allocator {
 public:
  void* Allocate(size_t bytes_size) override {
    return alloc_.allocate(bytes_size);
  }
  void Deallocate(void* ptr, size_t bytes_size) override {
    return alloc_.deallocate(static_cast<char*>(ptr), bytes_size);
  }
  const platform::Place& place() const override { return place_; }

 private:
  platform::Place place_{platform::CPUPlace()};
  std::allocator<char> alloc_;
};

template <typename T>
struct CustomAllocator {
  using value_type = T;
  using Allocator = experimental::framework::Allocator;

  explicit CustomAllocator(const std::shared_ptr<Allocator> a) noexcept
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
  std::shared_ptr<experimental::framework::Allocator> alloc_;
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
}  // namespace framework
}  // namespace experimental
}  // namespace paddle
