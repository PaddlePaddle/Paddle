/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/utils/test_macros.h"

namespace phi {

/// \brief The ExtendedTensor is a interface for custom designed class.
/// If you want to pass some self-designed data as input/output to kernels,
/// you can inherit from this class to store your self-designed data.
class TEST_API ExtendedTensor : public TensorBase {
 public:
  ExtendedTensor() = default;
  virtual ~ExtendedTensor() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "ExtendedTensor"; }

  int64_t numel() const override;

  const DDim& dims() const override;

  const Place& place() const override;

  DataType dtype() const override;

  DataLayout layout() const override;

  bool valid() const override;

  bool has_allocation() const override;

  bool initialized() const override;

  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;
};

}  // namespace phi
