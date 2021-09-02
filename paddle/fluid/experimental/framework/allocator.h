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

#include <cstdint>
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace experimental {
namespace framework {

/// \brief Encapsulates strategies for access/addressing, allocation/
/// deallocation and construction/destruction of objects.
class Allocator {
 public:
  /// \brief Default destructor.
  virtual ~Allocator() = default;

  /// \brief Allocates storage suitable for an array object of n bytes
  /// and creates the array, but does not construct array elements.
  /// May throw exceptions.
  /// \param bytes_size The number of bytes to allocate.
  /// \return The first address allocated.
  virtual void* Allocate(size_t bytes_size) = 0;

  /// \brief Deallocates storage pointed to ptr, which must be a value
  /// returned by a previous call to allocate that has not been
  /// invalidated by an intervening call to deallocate. The bytes_size
  /// must match the value previously passed to allocate.
  /// \param ptr The first address to deallocate.
  /// \param bytes_size The number of bytes to deallocate.
  virtual void Deallocate(void* ptr, size_t bytes_size) = 0;

  /// \brief Get the place value of the allocator and the allocation.
  /// \return The place value of the allocator and the allocation.
  virtual const platform::Place& place() const = 0;
};

inline void* Allocate(const std::shared_ptr<Allocator>& a, size_t n) {
  CHECK(a);
  return a->Allocate(n);
}

}  // namespace framework
}  // namespace experimental
}  // namespace paddle
