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

  explicit Storage(Allocation&& data) : data_(std::move(data)) {}

  virtual ~Storage() = default;

  /// \brief Get the mutable data pointer of the storage.
  /// This function is set to inline to improve performance.
  /// \return The mutable data pointer of the storage.
  void* data() const noexcept { return data_.operator->(); }

  virtual void Clear() = 0;

  virtual size_t size() const = 0;
  virtual const Place& place() const = 0;
  virtual bool OwnsMemory() const = 0;
  virtual void Realloc(size_t n) = 0;

 protected:
  Allocation data_;
};

class TensorStorage : public Storage {
 public:
  using Place = paddle::platform::Place;

  explicit TensorStorage(const std::shared_ptr<Allocator>& a) : alloc_(a) {}
  TensorStorage(const std::shared_ptr<Allocator>& a, size_t size)
      : Storage(Allocate(a, size)), alloc_(a), size_(size) {}

  ~TensorStorage() = default;

  static const char* name() { return "TensorStorage"; }

  void Realloc(size_t size) override;

  size_t size() const noexcept override { return size_; }

  void Clear() override {
    data_.Clear();
    size_ = 0;
  }

  const Place& place() const override { return data_.place(); }
  bool OwnsMemory() const noexcept override { return true; }
  const std::shared_ptr<Allocator>& allocator() const noexcept {
    return alloc_;
  }

 private:
  const std::shared_ptr<Allocator> alloc_;
  int64_t size_{0};
};

}  // namespace pten
