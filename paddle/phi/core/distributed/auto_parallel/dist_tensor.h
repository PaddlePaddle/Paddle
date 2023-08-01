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

#include <memory>
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace distributed {

namespace auto_parallel {
class TensorDistAttr;
}
using auto_parallel::TensorDistAttr;

class DistTensor final
    : public phi::TensorBase,
      public phi::TypeInfoTraits<phi::TensorBase, DistTensor> {
 public:
  /// \brief Construct a dist tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dist tensor.
  DistTensor(Allocator* a,
             const DenseTensorMeta& meta,
             const std::shared_ptr<TensorDistAttr>& dist_attr)
      : meta_(meta), dist_attr_(dist_attr) {
    value_ = std::make_unique<DenseTensor>(a, meta);
  }

  DistTensor(Allocator* a,
             DenseTensorMeta&& meta,
             const std::shared_ptr<TensorDistAttr>& dist_attr)
      : meta_(std::move(meta)), dist_attr_(dist_attr) {
    value_ = std::make_unique<DenseTensor>(a, meta);
  }

  DistTensor(const std::shared_ptr<phi::Allocation>& holder,
             const DenseTensorMeta& meta,
             const std::shared_ptr<TensorDistAttr>& dist_attr)
      : meta_(meta), dist_attr_(dist_attr) {
    value_ = std::make_unique<DenseTensor>(holder, meta);
  }

  DistTensor(const std::shared_ptr<phi::DenseTensor>& dense_tensor,
             const std::shared_ptr<TensorDistAttr>& dist_attr)
      : dist_attr_(dist_attr) {
    value_ = std::make_unique<DenseTensor>(*dense_tensor);
    set_meta(dense_tensor->meta());
  }

  ~DistTensor() = default;

  static const char* name() { return "DistTensor"; }

  const DenseTensor& value() const { return *value_; }

  DenseTensor* mutable_value() { return value_.get(); }

  const std::shared_ptr<TensorDistAttr>& dist_attr() const {
    return dist_attr_;
  }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override;

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const override { return meta_.dims; }

  /// \brief Test whether the storage is allocated.
  /// \return Whether the storage is allocated.
  bool initialized() const override {
    return value_ && value_->holder_ && value_->holder_->ptr();
  }

  bool defined() const { return value_ && value_->holder_; }

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const override { return meta_.valid(); }

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const override { return meta_.dtype; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const override { return meta_.layout; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override;

  const DenseTensorMeta& meta() const noexcept { return meta_; }

  /// \brief Sets the meta information of the tensor. Only when the original
  /// attribute of Tensor is incomplete, can it be reset.
  /// \param meta The meta information of the tensor.
  void set_meta(DenseTensorMeta&& meta);

  void set_meta(const DenseTensorMeta& meta);

 private:
  DenseTensorMeta meta_;
  std::shared_ptr<TensorDistAttr> dist_attr_{nullptr};
  std::unique_ptr<DenseTensor> value_{nullptr};
};

}  // namespace distributed
}  // namespace phi
