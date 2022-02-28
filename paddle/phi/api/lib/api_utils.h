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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/selected_rows.h"

namespace paddle {
namespace experimental {

/* ------------------ for input ----------------------- */

inline std::shared_ptr<phi::DenseTensor> TensorToDenseTensor(
    const Tensor& tensor) {
  return std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl());
}

inline std::shared_ptr<phi::DenseTensor> TensorToDenseTensor(
    const paddle::optional<Tensor>& tensor) {
  if (tensor) {
    return std::dynamic_pointer_cast<phi::DenseTensor>(tensor->impl());
  }
  return nullptr;
}

inline std::unique_ptr<std::vector<phi::DenseTensor>> TensorToDenseTensor(
    const std::vector<Tensor>& tensors) {
  auto pt_tensors = std::make_unique<std::vector<phi::DenseTensor>>();
  pt_tensors->reserve(tensors.size());

  for (const auto& t : tensors) {
    pt_tensors->push_back(
        *std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
  }

  return std::move(pt_tensors);
}

inline std::shared_ptr<phi::SelectedRows> TensorToSelectedRows(
    const Tensor& tensor) {
  return std::dynamic_pointer_cast<phi::SelectedRows>(tensor.impl());
}

inline std::shared_ptr<phi::SelectedRows> TensorToSelectedRows(
    const paddle::optional<Tensor>& tensor) {
  if (tensor) {
    return std::dynamic_pointer_cast<phi::SelectedRows>(tensor->impl());
  }
  return nullptr;
}

/* ----------------- for infer_meta --------------------- */

inline phi::MetaTensor MakeMetaTensor(const phi::DenseTensor& tensor) {
  return phi::MetaTensor(tensor);
}

inline paddle::optional<phi::MetaTensor> MakeMetaTensor(
    const paddle::optional<const phi::DenseTensor&>& tensor) {
  if (tensor) {
    return {phi::MetaTensor(*tensor)};
  }
  return {paddle::none};
}

inline std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<phi::DenseTensor>& tensors) {
  std::vector<phi::MetaTensor> meta_tensors;
  meta_tensors.reserve(tensors.size());
  for (const auto& t : tensors) {
    meta_tensors.emplace_back(t);
  }
  return meta_tensors;
}

inline phi::MetaTensor MakeMetaTensor(const phi::SelectedRows& tensor) {
  return phi::MetaTensor(tensor);
}

inline paddle::optional<phi::MetaTensor> MakeMetaTensor(
    const paddle::optional<const phi::SelectedRows&>& tensor) {
  if (tensor) {
    return {phi::MetaTensor(*tensor)};
  }
  return {paddle::none};
}

/* ------------------ for output ----------------------- */

inline phi::DenseTensor* SetKernelOutput(Backend backend, Tensor* out) {
  if (!out->initialized()) {
    auto dense_tensor = std::make_shared<phi::DenseTensor>(
        phi::make_intrusive<SharedStorage>(phi::TransToPtenPlace(backend)),
        phi::DenseTensorMeta());
    out->set_impl(dense_tensor);
    return dense_tensor.get();
  }
  return static_cast<phi::DenseTensor*>(out->impl().get());
}

inline std::vector<phi::DenseTensor*> SetKernelOutput(
    size_t out_size, Backend backend, std::vector<Tensor>* out) {
  out->reserve(out_size);
  std::vector<phi::DenseTensor*> results(out_size);
  for (size_t i = 0; i < out_size; ++i) {
    auto tensor_ptr = std::make_shared<phi::DenseTensor>(
        phi::make_intrusive<SharedStorage>(phi::TransToPtenPlace(backend)),
        phi::DenseTensorMeta());
    results[i] = tensor_ptr.get();
    out->emplace_back();
    out->back().set_impl(tensor_ptr);
  }
  return results;
}

inline phi::SelectedRows* SetSelectedRowsKernelOutput(Backend backend,
                                                      Tensor* out) {
  if (!out->initialized()) {
    auto select_rows = std::make_shared<phi::SelectedRows>();
    out->set_impl(select_rows);
    return select_rows.get();
  }
  return static_cast<phi::SelectedRows*>(out->impl().get());
}

}  // namespace experimental
}  // namespace paddle
