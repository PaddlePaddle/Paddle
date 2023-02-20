// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/stream.h"

namespace phi {

struct MemoryInterface {
  /**
   * @brief Allocate a unique allocation.
   *
   * @param[phi::Place] place     The target place that will be allocated
   * @param[size_t]     size      memory size
   */
  Allocator::AllocationPtr (*alloc)(const phi::Place& place, size_t size);

  /**
   * @brief Allocate a unique allocation.
   *
   * @param[phi::Place] place     The target gpu place that will be allocated
   * @param[size_t]     size      memory size
   * @param[phi::Stream]stream    the stream that is used for allocator
   */

  Allocator::AllocationPtr (*alloc_with_stream)(const phi::GPUPlace& place,
                                                size_t size,
                                                const phi::Stream& stream);

  /**
   * @brief Allocate a shared allocation.
   *
   * @param[phi::Place] place     The target place that will be allocated
   * @param[size_t]     size      memory size
   */
  std::shared_ptr<Allocation> (*alloc_shared)(const phi::Place& place,
                                              size_t size);

  /**
   * @brief Allocate a shared allocation.
   *
   * @param[phi::Place] place     The target place that will be allocated
   * @param[size_t]     size      memory size
   * @param[phi::Stream]stream    the stream that is used for allocator
   */
  std::shared_ptr<Allocation> (*alloc_shared_with_stream)(
      const phi::Place& place, size_t size, const phi::Stream& stream);

  /**
   * @brief whether the allocation is in the stream
   *
   * @param[Allocation] allocation  the allocation to check
   * @param[phi::Stream]stream      the device's stream
   */
  bool (*in_same_stream)(const std::shared_ptr<Allocation>& allocation,
                         const phi::Stream& stream);

  /**
   * @brief free allocation
   *
   * @param[Allocation] allocation  the allocation to be freed
   */
  void (*allocation_deleter)(Allocation* allocation);
};

class MemoryUtils {
 public:
  static MemoryUtils& Instance() {
    static MemoryUtils g_memory_utils;
    return g_memory_utils;
  }

  void Init(std::unique_ptr<MemoryInterface> memory_method) {
    memory_method_ = std::move(memory_method);
  }

  Allocator::AllocationPtr Alloc(const phi::GPUPlace& place,
                                 size_t size,
                                 const phi::Stream& stream) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(memory_method_->alloc_with_stream,
                      nullptr,
                      phi::errors::Unavailable(
                          "alloc_with_stream method in memory_method_ is not "
                          "initiazed yet. You need init it first."));
    return memory_method_->alloc_with_stream(place, size, stream);
  }

  Allocator::AllocationPtr Alloc(const phi::Place& place, size_t size) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->alloc,
        nullptr,
        phi::errors::Unavailable("alloc method in memory_method_ is not "
                                 "initiazed yet. You need init it first."));
    return memory_method_->alloc(place, size);
  }

  std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                          size_t size,
                                          const phi::Stream& stream) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(memory_method_->alloc_shared_with_stream,
                      nullptr,
                      phi::errors::Unavailable(
                          "alloc_shared_with_stream method in memory_method_ "
                          "is not initiazed yet. You need init it first."));
    return memory_method_->alloc_shared_with_stream(place, size, stream);
  }

  std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                          size_t size) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->alloc_shared,
        nullptr,
        phi::errors::Unavailable("alloc_shared method in memory_method_ is not "
                                 "initiazed yet. You need init it first."));
    return memory_method_->alloc_shared(place, size);
  }

  bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                    const phi::Stream& stream) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->in_same_stream,
        nullptr,
        phi::errors::Unavailable("in_same_stream method in memory_method_ is "
                                 "not initiazed yet. You need init it first."));
    return memory_method_->in_same_stream(allocation, stream);
  }

  void AllocationDeleter(Allocation* allocation) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(memory_method_->allocation_deleter,
                      nullptr,
                      phi::errors::Unavailable(
                          "allocation_deleter method in memory_method_ is not "
                          "initiazed yet. You need init it first."));
    return memory_method_->allocation_deleter(allocation);
  }

  void CheckMemoryMethod() {
    PADDLE_ENFORCE_NE(
        memory_method_.get(),
        nullptr,
        phi::errors::Unavailable("memory_method_ in MemoryUtils is not "
                                 "initiazed yet. You need init it first."));
  }

 private:
  MemoryUtils() = default;

  std::unique_ptr<MemoryInterface> memory_method_ = nullptr;

  DISABLE_COPY_AND_ASSIGN(MemoryUtils);
};

/*
  NOTE(YuanRisheng) Why should we add the following code?
  We need this because MemoryUtils::instance() is a singleton object and we
  don't recommend using singleton object in kernels. So, we wrap it using a
  function and if we delete this singleton object in future, it will be easy to
  change code.
*/

namespace memory_utils {

Allocator::AllocationPtr Alloc(const phi::GPUPlace& place,
                               size_t size,
                               const phi::Stream& stream);

Allocator::AllocationPtr Alloc(const phi::Place& place, size_t size);

std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                        size_t size,
                                        const phi::Stream& stream);

std::shared_ptr<Allocation> AllocShared(const phi::Place& place, size_t size);

bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                  const phi::Stream& stream);

void AllocationDeleter(Allocation* allocation);

}  // namespace memory_utils

}  // namespace phi
