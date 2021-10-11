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

#include <vector>

#include "paddle/tcmpt/core//dense_tensor.h"
#include "paddle/tcmpt/core/tensor_base.h"

namespace paddle {
namespace tcmpt {

class SelectedRows : public TensorBase,
                     public TypeInfoTraits<TensorBase, SelectedRows> {
 public:
  SelectedRows() = default;

  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "SelectedRows"; }

  /// \brief Returns the number of elements contained in tensor.
  /// \return The number of elements contained in tensor.
  int64_t numel() const { return {}; }

  /// \brief Returns the dims of the tensor.
  /// \return The dims of the tensor.
  const DDim& dims() const noexcept { return dims_; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType data_type() const noexcept { return {}; }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const noexcept { return {}; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const platform::Place& place() const { return place_; }

 private:
  platform::Place place_;
  DDim dims_;
};

}  // namespace tcmpt
}  // namespace paddle
