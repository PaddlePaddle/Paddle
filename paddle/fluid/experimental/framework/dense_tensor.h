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

#include "boost/smart_ptr/intrusive_ptr.hpp"

#include "paddle/fluid/experimental/framework/allocator.h"
#include "paddle/fluid/experimental/framework/data_type.h"
#include "paddle/fluid/experimental/framework/storage.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace experimental {
namespace framework {

using DDim = paddle::framework::DDim;

struct DenseTensorMeta {
  DenseTensorMeta() = default;
  DenseTensorMeta(DataType type, const DDim& dims) : dims(dims), type(type) {}
  DenseTensorMeta(DataType type, const DDim& dims, DataLayout layout)
      : dims(dims), type(type), layout(layout) {}
  DenseTensorMeta(DataType type, const DDim& dims, DataLayout layout,
                  const std::vector<std::vector<size_t>>& lod)
      : dims(dims), type(type), layout(layout), lod(lod) {}

  bool valid() const noexcept {
    bool valid{true};
    valid = valid && (type != DataType::INVALID);
    valid = valid && (layout != DataLayout::Undef);
    valid = valid && (is_scalar || product(dims));
    return valid;
  }

  bool is_scalar{false};
  DDim dims;
  DataType type{DataType::FLOAT32};
  DataLayout layout{DataLayout::NCHW};
  std::vector<std::vector<size_t>> lod;
};

class DenseTensor {
 public:
  DenseTensor() = default;

  DenseTensor(const std::shared_ptr<Allocator>& a, const DenseTensorMeta& meta)
      : meta_(meta),
        storage_(new TensorStorage(a, SizeOf(data_type()) * numel())) {}

  DenseTensor(const std::shared_ptr<Allocator>& a, DenseTensorMeta&& meta)
      : meta_(std::move(meta)),
        storage_(new TensorStorage(a, SizeOf(data_type()) * numel())) {}

  DenseTensor(boost::intrusive_ptr<Storage>&& storage,
              const DenseTensorMeta& meta)
      : meta_(meta), storage_(std::move(storage)) {}

 public:
  DenseTensor(const DenseTensor& other) = delete;
  DenseTensor(DenseTensor&& other) = delete;
  virtual ~DenseTensor() = default;

 public:
  int64_t numel() const {
    if (meta_.is_scalar) {
      return 1;
    }
    return product(meta_.dims);
  }
  const DDim& dims() const noexcept { return meta_.dims; }
  DataType data_type() const noexcept { return meta_.type; }
  DataLayout layout() const noexcept { return meta_.layout; }
  const platform::Place& place() const { return storage_->place(); }
  bool initialized() const noexcept { return storage_; }
  bool SharesStorageWith(const DenseTensor& b) const {
    return storage_->root_storage() == b.storage_->root_storage();
  }

  template <typename T>
  T* mutable_data(size_t request_bytes) {
    CHECK(meta_.type == DataTypeTrait<T>::DataType());
    size_t bytes = numel() * SizeOf(data_type());
    if (request_bytes) {
      CHECK_GE(request_bytes, bytes);
      bytes = request_bytes;
    }
    if (storage_->size() < bytes) {
      CHECK(dynamic_cast<TensorStorage*>(storage_.get()));
      // storage_.reset(std::make_shared<Storage>(a, bytes));
    }
    return reinterpret_cast<T*>(storage_->data());
  }

  template <typename T>
  const T* data() const {
    CHECK(meta_.type == DataTypeTrait<T>::DataType());
    return reinterpret_cast<const T*>(storage_->data());
  }

  void Resize(const DDim& dims) noexcept { meta_.dims = dims; }

  size_t memory_size() const { return storage_->size(); }

  void check_memory_size() const {
    size_t bytes = numel() * SizeOf(data_type());
    CHECK_GE(memory_size(), bytes);
  }

  boost::intrusive_ptr<Storage> release() { return std::move(storage_); }

 private:
  DenseTensorMeta meta_;
  boost::intrusive_ptr<Storage> storage_;
};

}  // namespace framework
}  // namespace experimental
}  // namespace paddle
