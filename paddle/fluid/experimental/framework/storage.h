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
#include "paddle/fluid/experimental/framework/utils/intrusive_ptr.h"
#include "paddle/fluid/experimental/framework/utils/intrusive_ref_counter.h"

#include "paddle/fluid/experimental/framework/allocator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace experimental {
namespace framework {

/// \brief The inferface of ontiguous storage used for the dense tensor.
/// It should be used in conjunction with the intrusive pointer. We prohibit
/// all default copy operations to ensure the integrity of the package.
class StorageInterface : public intrusive_ref_counter<StorageInterface> {
 public:
  StorageInterface() = default;
  explicit StorageInterface(void* data_ptr) noexcept : data_(data_ptr) {}

  virtual ~StorageInterface() = default;

  /// \brief Get the mutable data pointer of the storage.
  /// This function is set to inline to improve performance.
  /// \return The mutable data pointer of the storage.
  void* data() const noexcept { return data_; }

  virtual size_t size() const = 0;
  virtual const platform::Place& place() const = 0;
  virtual bool OwnsMemory() const = 0;
  virtual void Realloc(size_t n) = 0;

  StorageInterface(const StorageInterface&) = delete;
  StorageInterface& operator=(const StorageInterface&) = delete;

 protected:
  void* data_{nullptr};
};

class Storage : public StorageInterface {
 public:
  explicit Storage(const std::shared_ptr<Allocator>& a) : alloc_(a) {}
  Storage(const std::shared_ptr<Allocator>& a, size_t size)
      : StorageInterface(Allocate(a, size)), alloc_(a), size_(size) {}

  ~Storage() { alloc_->Deallocate(data(), size_); }

  void Realloc(size_t size) override;

  size_t size() const noexcept override { return size_; }
  const platform::Place& place() const override { return alloc_->place(); }
  bool OwnsMemory() const noexcept override { return true; }
  const std::shared_ptr<Allocator>& allocator() const noexcept {
    return alloc_;
  }

 private:
  const std::shared_ptr<Allocator> alloc_;
  int64_t size_{0};
};

class ExternalStorage : public StorageInterface {
 public:
  ExternalStorage(void* ptr, size_t size, const platform::Place& place);
  ExternalStorage(const intrusive_ptr<Storage>& root, size_t delta,
                  size_t size);

  void Realloc(size_t n) override {
    PADDLE_THROW(platform::errors::Unavailable(
        "The external shared storage cannot be reallocated."));
  }

  size_t size() const noexcept override { return size_; }
  const platform::Place& place() const override { return place_; }
  bool OwnsMemory() const noexcept override { return false; }

 private:
  const int64_t size_{0};
  const platform::Place place_;
};

}  // namespace framework
}  // namespace experimental
}  // namespace paddle
