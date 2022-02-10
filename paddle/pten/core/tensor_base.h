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

#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/common/place.h"
#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/ddim.h"
#include "paddle/pten/core/utils/type_registry.h"

namespace pten {

class TensorBase {
 public:
  using DDim = pten::framework::DDim;

  virtual ~TensorBase() = default;

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  virtual int64_t numel() const = 0;

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  virtual const DDim& dims() const = 0;

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  virtual DataType dtype() const = 0;

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  virtual DataLayout layout() const = 0;

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  virtual const Place& place() const = 0;

  /// \brief Test whether the metadata is valid.
  /// \return Whether the metadata is valid.
  virtual bool valid() const = 0;

  /// \brief Test whether the storage is allocated.
  /// return Whether the storage is allocated.
  virtual bool initialized() const = 0;

  // TODO(Aurelius84): This interface is under intermediate state now.
  // We will remove DataType argument in the future. Please DO NOT
  // rely on Datatype to much when design and implement other feature.

  /// \brief Allocate memory with requested size from allocator.
  /// \return The mutable data pointer value of type T.
  virtual void* AllocateFrom(Allocator* allocator,
                             DataType dtype,
                             size_t requested_size = 0) = 0;

  /// \brief Return the type information of the derived class to support
  /// safely downcast in non-rtti environment.
  /// return The type information of the derived class.
  TypeInfo<TensorBase> type_info() const { return type_info_; }

 private:
  template <typename T, typename U>
  friend class TypeInfoTraits;
  TypeInfo<TensorBase> type_info_{TypeInfo<TensorBase>::kUnknownType};
};

}  // namespace pten
