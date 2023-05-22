/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/storage_properties.h"
#include "paddle/phi/core/stream.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

class DistAttr {};

class DenseTensor;

class DistTensor : public TensorBase,
                   public TypeInfoTraits<TensorBase, DistTensor> {
 public:
  using DDim = phi::DDim;

  static const char* name() { return "DistTensor"; }

  DistTensor();
  /// \brief Construct a dist tensor and allocate space.
  /// \param a The allocator used to allocate space.
  /// \param meta The meta data of dense tensor.
  DistTensor(Allocator* a, const DenseTensorMeta& meta);

  DistTensor(const std::shared_ptr<phi::Allocation>& holder,
             const DenseTensorMeta& meta);

  explicit DistTensor(const std::shared_ptr<phi::DenseTensor>& dense_tensor);

  ~DistTensor() = default;

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const override;

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const override;

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const override;

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const override;

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override;

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  bool valid() const override;

  /// \brief Test whether the storage is allocated.
  /// \return Whether the storage is allocated.
  bool initialized() const override;
  // TODO(Aurelius84): This interface is under intermediate state now.
  // We will remove DataType argument in the future. Please DO NOT
  // rely on Datatype too much when designing and implementing other features.

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

  const DenseTensorMeta& meta() const noexcept;

  /// \brief Sets the meta information of the tensor. Only when the original
  /// attribute of Tensor is incomplete, can it be reset.
  /// \param meta The meta information of the tensor.
  void set_meta(DenseTensorMeta&& meta);

  void set_meta(const DenseTensorMeta& meta);

  const DistAttr& get_dist_attr();

  const std::shared_ptr<DenseTensor>& local_tensor() const;

 private:
  // dist attribute
  std::unique_ptr<DistAttr> dist_attr_;
  // local shard of the tensor
  std::shared_ptr<DenseTensor> local_tensor_;
};

}  // namespace phi
