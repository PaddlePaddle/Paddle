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

#include <cstddef>

#include "boost/intrusive_ptr.hpp"
#include "paddle/pten/core/utils/intrusive_ptr.h"
#include "paddle/pten/core/utils/intrusive_ref_counter.h"
#include "paddle/pten/core/utils/type_info.h"

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/pten/core/allocator.h"

namespace pten {

/// \brief The interface of contiguous storage used for the dense tensor.
/// It should be used in conjunction with the intrusive pointer. We prohibit
/// all default copy operations to ensure the integrity of the package.
class Storage : public intrusive_ref_counter<Storage> {
 public:
  using Place = paddle::platform::Place;
  Storage() = default;
  Storage(const Storage&) = delete;

  /* --------- shared_ptr<Allocation> -------- */
  // Initialize a Storage with unique Allocation
  explicit Storage(std::shared_ptr<paddle::memory::Allocation>&& data)
      : data_(std::move(data)) {}

  // Initialize a Storage shareing Allocation with another storage
  explicit Storage(const std::shared_ptr<paddle::memory::Allocation>& data)
      : data_(data) {}

  void* data() const {
    return data_ ? reinterpret_cast<void*>(
                       reinterpret_cast<uintptr_t>(data_->ptr()) + offset_)
                 : nullptr;
  }

  const std::shared_ptr<paddle::memory::Allocation> data_shared() const {
    return data_;
  }

  virtual void ReallocShared(size_t n) {
    PADDLE_THROW(paddle::platform::errors::Unimplemented(
        "ReallocShared has not been overrided by the current Storage"));
  }
  /* --------- shared_ptr<Allocation> -------- */

  virtual ~Storage() = default;

  virtual void Clear() = 0;

  virtual size_t size() const = 0;
  virtual const Place& place() const = 0;
  virtual bool OwnsMemory() const = 0;
  virtual void Realloc(size_t n) = 0;

 protected:
  size_t offset_{0};
  std::shared_ptr<paddle::memory::Allocation> data_;
};

class TensorStorage : public Storage {
 public:
  using Place = paddle::platform::Place;

  explicit TensorStorage(const std::shared_ptr<Allocator>& a) : alloc_(a) {}

  TensorStorage(const std::shared_ptr<Allocator>& a, size_t size)
      : Storage(paddle::memory::AllocShared(a->place(), size)), alloc_(a) {
    size_ = data_->size();
  }

  void Clear() override {
    data_ = nullptr;
    size_ = 0;
    offset_ = 0;
  }

  void Realloc(size_t size) override;

  ~TensorStorage() = default;

  static const char* name() { return "TensorStorage"; }

  size_t size() const noexcept override { return size_; }

  const Place& place() const override {
    if (!data_ && !alloc_) {
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unable to visit place: either data_ or alloc_ has to be initialized "
          "first."));
    }
    if (data_) {
      return data_->place();
    }
    return alloc_->place();
  }

  bool OwnsMemory() const noexcept override { return true; }
  const std::shared_ptr<Allocator>& allocator() const noexcept {
    return alloc_;
  }

 private:
  const std::shared_ptr<Allocator> alloc_;
  int64_t size_{0};
};

}  // namespace pten
