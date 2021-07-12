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

#include "paddle/pten/core/tensor_impl_if.h"
#include "paddle/pten/core/tensor_meta.h"

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
 * The implementation of general Tensor (For CPU, CUDA, HIP, etc.),
 * contains a pointer to Allocation and a series of descriptive metadata
 * required by Tensor.
 *
 * BaseTensor is still a base class, it may have mutiple inherited classes,
 * such as LoDTensor, SelectedRows, etc. The memory layout
 * of these inherited classes is consistent with the basic BaseTensor, except
 * that a small number of members are added to further specialize the
 * description of the tensor. For example, LoDTensor adds LoD information,
 * and SelectedRows adds rows and height information.
 * If the memory layout is different, it cannot be described based on the
 * general Allocation, and it needs to be directly inherited from
 * TensorImplInterface.
 *
 */
class BaseTensor : public TensorImplInterface {
 public:
  // Not allowed to initialize a tensor without descriptive metadata
  BaseTensor() = delete;

  BaseTensor(const BaseTensor&) = delete;
  BaseTensor& operator=(const BaseTensor&) = delete;
  BaseTensor(BaseTensor&&) = delete;
  BaseTensor& operator=(BaseTensor&&) = delete;

  /**
   * If we still malloc memory by mutable_data,
   * the BaseTensor doesn't need complicated constructor.
   *
   * Note: Tensor objects lacking meta information are not allowed to exist.
   */
  explicit BaseTensor(TensorMeta meta);

  ~BaseTensor() override {}

  /**
   * Most of Tensor's methods need to have corresponding implementations
   * in BaseTensor
   */
  int64_t numel() const override;

  DDim dims() const override;

  void resize(const DDim& dims) override;

  DataType type() const override;

  Layout layout() const override;

  Place place() const override;

  Backend backend() const override;

  const void* data() const override;

  void* mutable_data() override;

  bool initialized() const override;

  /**
   * using base class template methods.
   */
  using TensorImplInterface::data;
  using TensorImplInterface::mutable_data;

  // For non-API interfaces, we still follow the C++ code style
  void ShareAllocation(const std::shared_ptr<Allocation>& memory);

  Place GetPlaceByBackend() const;

  size_t MemorySize() const;

  void CheckMemorySize() const;

  std::shared_ptr<Allocation> MoveMemory();

 private:
  // The actual Tensor storage holder
  std::shared_ptr<Allocation> memory_;
  // The Tensor meta data
  TensorMeta meta_;
};

}  // namespace pt
