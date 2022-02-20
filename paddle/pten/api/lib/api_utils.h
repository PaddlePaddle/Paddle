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

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/compat/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/meta_tensor.h"
#include "paddle/pten/core/selected_rows.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {

/* ------------------ for input ----------------------- */

inline std::shared_ptr<pten::DenseTensor> TensorToDenseTensor(
    const Tensor& tensor) {
  return std::dynamic_pointer_cast<pten::DenseTensor>(tensor.impl());
}

inline paddle::optional<const pten::DenseTensor&> TensorToDenseTensor(
    const paddle::optional<Tensor>& tensor) {
  if (tensor) {
    return {*std::dynamic_pointer_cast<pten::DenseTensor>(tensor->impl())};
  }
  return paddle::none;
}

inline std::unique_ptr<std::vector<pten::DenseTensor>> TensorToDenseTensor(
    const std::vector<Tensor>& tensors) {
  auto pt_tensors = std::make_unique<std::vector<pten::DenseTensor>>();
  pt_tensors->reserve(tensors.size());

  for (const auto& t : tensors) {
    pt_tensors->push_back(
        *std::dynamic_pointer_cast<pten::DenseTensor>(t.impl()));
  }

  return std::move(pt_tensors);
}

inline std::shared_ptr<pten::SelectedRows> TensorToSelectedRows(
    const Tensor& tensor) {
  return std::dynamic_pointer_cast<pten::SelectedRows>(tensor.impl());
}

inline paddle::optional<const pten::SelectedRows&> TensorToSelectedRows(
    const paddle::optional<Tensor>& tensor) {
  if (tensor) {
    return {*std::dynamic_pointer_cast<pten::SelectedRows>(tensor->impl())};
  }
  return paddle::none;
}

/* ----------------- for infer_meta --------------------- */

inline pten::MetaTensor MakeMetaTensor(const pten::DenseTensor& tensor) {
  return pten::MetaTensor(tensor);
}

inline paddle::optional<pten::MetaTensor> MakeMetaTensor(
    const paddle::optional<const pten::DenseTensor&>& tensor) {
  if (tensor) {
    return pten::MetaTensor(*tensor);
  }
  return paddle::none;
}

inline std::vector<pten::MetaTensor> MakeMetaTensor(
    const std::vector<pten::DenseTensor>& tensors) {
  std::vector<pten::MetaTensor> meta_tensors;
  meta_tensors.reserve(tensors.size());
  for (const auto& t : tensors) {
    meta_tensors.emplace_back(t);
  }
  return meta_tensors;
}

inline pten::MetaTensor MakeMetaTensor(const pten::SelectedRows& tensor) {
  return pten::MetaTensor(tensor);
}

inline paddle::optional<pten::MetaTensor> MakeMetaTensor(
    const paddle::optional<const pten::SelectedRows&>& tensor) {
  if (tensor) {
    return pten::MetaTensor(*tensor);
  }
  return paddle::none;
}

/* ------------------ for output ----------------------- */

inline pten::DenseTensor* SetKernelOutput(Backend backend, Tensor* out) {
  auto dense_tensor = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<SharedStorage>(pten::TransToPtenPlace(backend)),
      pten::DenseTensorMeta());
  out->set_impl(dense_tensor);
  return dense_tensor.get();
}

inline std::vector<pten::DenseTensor*> SetKernelOutput(
    size_t out_size, Backend backend, std::vector<Tensor>* out) {
  out->reserve(out_size);
  std::vector<pten::DenseTensor*> results(out_size);
  for (size_t i = 0; i < out_size; ++i) {
    auto tensor_ptr = std::make_shared<pten::DenseTensor>(
        pten::make_intrusive<SharedStorage>(pten::TransToPtenPlace(backend)),
        pten::DenseTensorMeta());
    results[i] = tensor_ptr.get();
    out->emplace_back();
    out->back().set_impl(tensor_ptr);
  }
  return results;
}

inline pten::SelectedRows* SetSelectedRowsKernelOutput(Backend backend,
                                                       Tensor* out) {
  auto select_rows = std::make_shared<pten::SelectedRows>();
  out->set_impl(select_rows);
  return select_rows.get();
}

}  // namespace experimental
}  // namespace paddle
