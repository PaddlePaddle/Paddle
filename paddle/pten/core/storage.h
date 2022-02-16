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
#include "paddle/pten/common/place.h"
#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/utils/intrusive_ptr.h"
#include "paddle/pten/core/utils/intrusive_ref_counter.h"
#include "paddle/pten/core/utils/type_info.h"

namespace pten {

/// \brief The interface of contiguous storage used for the dense tensor.
/// It should be used in conjunction with the intrusive pointer. We prohibit
/// all default copy operations to ensure the integrity of the package.
class Storage : public intrusive_ref_counter<Storage> {
 public:
  Storage() = default;
  Storage(const Storage&) = delete;

  /* @jim19930609: Following interfaces will be modified/replaced/removed
                   as soon as the new Allocation - Allocator design get
     finalized.
    */

  /*   --------- shared_ptr<Allocation> -------- */
  // Initialize a Storage with unique Allocation
  explicit Storage(std::shared_ptr<pten::Allocation>&& data)
      : data_(std::move(data)) {}

  // Initialize a Storage shareing Allocation with another storage
  explicit Storage(const std::shared_ptr<pten::Allocation>& data)
      : data_(data) {}

  void* data() const {
    return data_ ? reinterpret_cast<void*>(
                       reinterpret_cast<uintptr_t>(data_->ptr()))
                 : nullptr;
  }

  const std::shared_ptr<pten::Allocation>& data_shared() const { return data_; }

  virtual void set_data_shared(
      const std::shared_ptr<pten::Allocation>& holder) = 0;

  virtual std::shared_ptr<pten::Allocation>&& move_data_shared() = 0;

  virtual void ReallocShared(size_t n) {
    PADDLE_THROW(pten::errors::Unimplemented(
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
  std::shared_ptr<pten::Allocation> data_;
};

class TensorStorage : public Storage {
 public:
  explicit TensorStorage(Allocator* a) : alloc_(a) {}

  TensorStorage(Allocator* a, size_t size)
      : Storage(a->Allocate(size)), alloc_(a) {
    size_ = data_->size();
  }

  void Clear() override {
    data_ = nullptr;
    size_ = 0;
  }

  void Realloc(size_t size) override;

  ~TensorStorage() = default;

  static const char* name() { return "TensorStorage"; }

  size_t size() const noexcept override { return size_; }

  const Place& place() const override {
    if (!data_) {
      PADDLE_THROW(pten::errors::Unimplemented(
          "Unable to visit place: either data_ or alloc_ has to be initialized "
          "first."));
    }
    return data_->place();
  }

  bool OwnsMemory() const noexcept override { return true; }

  void set_data_shared(
      const std::shared_ptr<pten::Allocation>& holder) override {
    CHECK(holder);
    data_ = holder;
    size_ = holder->size();
  }

  std::shared_ptr<pten::Allocation>&& move_data_shared() override {
    size_ = 0;
    return std::move(data_);
  }

 private:
  Allocator* alloc_;
  int64_t size_{0};
};

}  // namespace pten
