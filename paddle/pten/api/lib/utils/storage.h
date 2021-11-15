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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/pten/core/storage.h"

namespace paddle {
namespace experimental {

class ExternalStorage : public pten::Storage {
 public:
  ExternalStorage(void* ptr, size_t size, const paddle::platform::Place& place);
  ExternalStorage(const pten::intrusive_ptr<pten::Storage>& root,
                  size_t delta,
                  size_t size);

  static const char* name() { return "ExternalStorage"; }

  void Realloc(size_t n) override {
    PADDLE_THROW(paddle::platform::errors::Unavailable(
        "The external shared storage cannot be reallocated."));
  }

  void Clear() override {
    data_.Clear();
    size_ = 0;
  }

  size_t size() const noexcept override { return size_; }
  const paddle::platform::Place& place() const override {
    return data_.place();
  }
  bool OwnsMemory() const noexcept override { return false; }

 private:
  int64_t size_{0};
};

class SharedStorage : public pten::Storage {
 public:
  explicit SharedStorage(
      const std::shared_ptr<paddle::memory::Allocation>& allocation,
      size_t offset)
      : allocation_(allocation) {
    CHECK(allocation);
    data_ = pten::Allocation(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(allocation->ptr()) +
                                offset),
        allocation->place());
    size_ = allocation->size();
  }

  // In order to be compatible with the original Tensor design and execution
  // system, we need to allow the uninitialized SharedStorage to exist,
  // and it can be removed after the compatibility phase is over in the future
  explicit SharedStorage(const paddle::platform::Place& place) {
    data_ = pten::Allocation(nullptr, place);
  }

  static const char* name() { return "SharedStorage"; }

  // In order to be compatible with the original Tensor design and execution
  // system, we need to allow the SharedStorage realloc,
  // and it can be removed after the compatibility phase is over in the future
  void Realloc(size_t n) override {
    if (data() != nullptr) {
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "The external shared storage cannot be reallocated."));
    }
    ResetAllocation(paddle::memory::AllocShared(place(), n), 0);
  }

  void Clear() override {
    data_.Clear();
    size_ = 0;
  }

  size_t size() const noexcept override { return size_; }
  const paddle::platform::Place& place() const override {
    return data_.place();
  }
  bool OwnsMemory() const noexcept override { return false; }

  const std::shared_ptr<paddle::memory::Allocation>& GetAllocation() {
    return allocation_;
  }

  // Temporary method: For compatible with fluid Tensor and improve performance
  void ResetAllocation(std::shared_ptr<paddle::memory::Allocation> allocation,
                       size_t offset) {
    allocation_ = allocation;
    data_ = pten::Allocation(
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(allocation->ptr()) +
                                offset),
        allocation->place());
    size_ = allocation->size();
  }

  // Temporary method: For compatible with fluid Tensor and improve performance
  void Reset() {
    allocation_.reset();
    data_.Clear();
    size_ = 0;
  }

 private:
  int64_t size_{0};
  std::shared_ptr<paddle::memory::Allocation> allocation_;
};

class TensorStorage : public paddle::memory::allocation::Allocation {
 public:
  explicit TensorStorage(pten::intrusive_ptr<pten::Storage> storage)
      : paddle::memory::allocation::Allocation(
            storage->data(), storage->size(), storage->place()),
        storage_(std::move(storage)) {}

 private:
  pten::intrusive_ptr<pten::Storage> storage_;
};

}  // namespace experimental
}  // namespace paddle
