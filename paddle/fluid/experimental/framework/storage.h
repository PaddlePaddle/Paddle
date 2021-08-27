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

class Storage : public intrusive_ref_counter<Storage> {
 public:
  explicit Storage(void* data_ptr) noexcept : data_(data_ptr) {}

  void* data() const noexcept { return data_; }
  virtual size_t size() const = 0;
  virtual Storage* root_storage() = 0;
  virtual const platform::Place& place() const = 0;
  virtual bool OwnsMemory() const { return true; }

 protected:
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  void* const data_;
};

class TensorStorage final : public Storage {
 public:
  TensorStorage(const std::shared_ptr<Allocator>& a, int64_t n)
      : Storage(Allocate(a, n)), alloc_(a), size_(n) {}

  ~TensorStorage() { alloc_->Deallocate(data_, size_); }

  size_t size() const noexcept override { return size_; }
  Storage* root_storage() noexcept override { return this; }
  const platform::Place& place() const override { return alloc_->place(); }
  bool OwnsMemory() const noexcept override { return true; }
  const std::shared_ptr<Allocator>& allocator() const noexcept {
    return alloc_;
  }

 private:
  const std::shared_ptr<Allocator> alloc_;
  int64_t size_;
};

class SubStorage final : public Storage {
 public:
  SubStorage(const intrusive_ptr<Storage>& root, int64_t delta, int64_t n)
      : Storage(static_cast<uint8_t*>(root->data()) + delta),
        root_(copy_intrusive(root)),
        size_(n) {
    CHECK_GT(n, 0);
    CHECK_LE(static_cast<size_t>(delta + n), root->size());
  }

  size_t size() const noexcept override { return size_; }
  Storage* root_storage() noexcept override { return root_.get(); }
  const platform::Place& place() const override { return root_->place(); }
  bool OwnsMemory() const noexcept override { return false; }

 private:
  const intrusive_ptr<Storage> root_;
  int64_t size_;
};

}  // namespace framework
}  // namespace experimental
}  // namespace paddle
