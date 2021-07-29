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

#include <memory>

#include "paddle/top/core/tensor_impl_if.h"
#include "paddle/top/core/tensor_meta.h"
#include "paddle/top/core/tensor_status.h"

namespace paddle {
namespace memory {
namespace allocation {
class Allocation;
}
}
}

namespace pt {

// TODO(chenweihang): Allocation still link to framework, Redesign and
// decoupled Allocation and Allocator?
using Allocation = paddle::memory::allocation::Allocation;

/**
 * The implementation of general Tensor (For CPU, CUDA, HIP, etc.), similar
 * to the Tensor in fluid, contains a pointer to Allocation and a series of
 * descriptive metadata and status required by Tensor.
 *
 * DenseTensor is still a base class, it may have inherited classes.
 *
 * The memory layout of these inherited classes is consistent with the
 * basic DenseTensor, except that a small number of members are added to
 * further specialize the description of the tensor.
 *
 * If the memory layout is different, it cannot be described based on the
 * general Allocation, and it needs to be directly inherited from
 * TensorImplInterface.
 */
class DenseTensor : public TensorImplInterface {
 public:
  // Not allowed to initialize a tensor without descriptive metadata
  DenseTensor() = delete;

  DenseTensor(const DenseTensor&) = delete;
  DenseTensor& operator=(const DenseTensor&) = delete;
  DenseTensor(DenseTensor&&) = delete;
  DenseTensor& operator=(DenseTensor&&) = delete;

  /**
   * If we still malloc memory by mutable_data,
   * the DenseTensor doesn't need complicated constructor.
   *
   * Note: Tensor objects lacking meta information are not allowed to exist.
   */
  explicit DenseTensor(std::unique_ptr<TensorMeta> meta,
                       std::unique_ptr<TensorStatus> status =
                           std::unique_ptr<TensorStatus>(new TensorStatus()))
      : meta_(std::move(meta)), status_(std::move(status)) {}

  ~DenseTensor() override {}

  int64_t numel() const override { return meta_->numel; }

  DDim dims() const override { return meta_->dims; }

  DataType type() const override { return meta_->type; }

  DataLayout layout() const override { return meta_->layout; }

  Place place() const override;

  Backend backend() const override { return meta_->backend; }

  bool initialized() const override { return allocation_ != nullptr; }

  /* member methods */

  const std::shared_ptr<Allocation>& allocation() const { return allocation_; }

  const TensorMeta& meta() const { return *meta_; }

  TensorMeta* mutable_meta() { return meta_.get(); }

  /* Data Access Methods */

  const void* data() const;

  void* mutable_data();

  template <typename T>
  const T* data() const {
    static_assert(std::is_pod<T>::value,
                  "T must be POD when call Tensor.data<T>().");
    return reinterpret_cast<const T*>(data());
  }

  // mutable_data does not hold arguments.
  // Before calling mutable_data, please make sure that Tensor has maintained
  // the correct meta and status.
  template <typename T>
  T* mutable_data() {
    static_assert(std::is_pod<T>::value,
                  "T must be POD when call Tensor.mutable_data<T>().");
    return reinterpret_cast<T*>(mutable_data());
  }

  // For non-API and non-member interfaces, we still follow the C++ code style?

  void Resize(const DDim& dims) { meta_->dims = dims; }

  void ShareAllocation(const std::shared_ptr<Allocation>& allocation);

  Place GetPlaceByBackend() const;

  size_t MemorySize() const;

  void CheckMemorySize() const;

  std::shared_ptr<Allocation> MoveMemory();

 private:
  // The actual Tensor storage holder
  std::shared_ptr<Allocation> allocation_;
  // The Tensor meta data
  std::unique_ptr<TensorMeta> meta_;
  // The Tensor status data
  std::unique_ptr<TensorStatus> status_;
};

}  // namespace pt
