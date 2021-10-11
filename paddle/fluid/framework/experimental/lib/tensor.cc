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

#include "paddle/fluid/framework/experimental/include/tensor.h"
#include "paddle/fluid/framework/experimental/lib/tensor_utils.h"
#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/selected_rows.h"

namespace paddle {
namespace experimental {

template <>
std::shared_ptr<void> make_tensor_impl(
    std::unique_ptr<tcmpt::TensorBase>&& impl) {
  return std::shared_ptr<void>(std::move(impl));
}

int64_t Tensor::numel() const { return Utils::GetImpl(*this)->numel(); }

const Tensor::Shape Tensor::shape() const {
  return framework::vectorize(Utils::GetImpl(*this)->dims());
}

DataType Tensor::data_type() const {
  return Utils::GetImpl(*this)->data_type();
}

DataLayout Tensor::layout() const { return Utils::GetImpl(*this)->layout(); }

template <typename T>
T* Tensor::data() const {
  return nullptr;
}

template <typename T>
T* Tensor::mutable_data() {
  return nullptr;
}

bool Tensor::valid() const { return Utils::GetImpl(*this)->valid(); }

bool Tensor::initialized() const {
  return Utils::GetImpl(*this)->initialized();
}

bool Tensor::is_dense_tensor() const {
  return tcmpt::TypeInfoTraits<tcmpt::TensorBase, tcmpt::DenseTensor>::classof(
      Utils::GetImpl(*this));
}

bool Tensor::is_selected_rows() const {
  return tcmpt::TypeInfoTraits<tcmpt::TensorBase, tcmpt::SelectedRows>::classof(
      Utils::GetImpl(*this));
}

bool Tensor::is_cpu() const {
  return platform::is_cpu_place(Utils::GetImpl(*this)->place());
}

bool Tensor::is_cuda() const {
  return platform::is_gpu_place(Utils::GetImpl(*this)->place());
}

Tensor Tensor::cpu() const { return {}; }

Tensor Tensor::cuda() const { return {}; }

}  // namespace experimental
}  // namespace paddle
