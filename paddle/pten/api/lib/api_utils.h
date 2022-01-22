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
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"

namespace paddle {
namespace experimental {

/* ------------------ for input ----------------------- */

inline std::shared_ptr<pten::DenseTensor> TensorToDenseTensor(
    const Tensor& tensor) {
  return std::dynamic_pointer_cast<pten::DenseTensor>(tensor.impl());
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

/* ----------------- for infer_meta --------------------- */

inline const pten::DenseTensorMeta& GetDenseTensorMeta(
    const pten::DenseTensor& tensor) {
  return tensor.meta();
}

inline std::vector<pten::DenseTensorMeta> GetDenseTensorMeta(
    const std::vector<pten::DenseTensor>& tensors) {
  std::vector<pten::DenseTensorMeta> metas;
  metas.reserve(tensors.size());
  for (const auto& t : tensors) {
    metas.push_back(t.meta());
  }
  return metas;
}

/* ------------------ for output ----------------------- */

inline pten::DenseTensor* SetKernelOutput(const pten::DenseTensorMeta& meta,
                                          Backend backend,
                                          Tensor* out) {
  auto dense_tensor = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<SharedStorage>(pten::TransToFluidPlace(backend)),
      meta);
  out->set_impl(dense_tensor);
  return dense_tensor.get();
}

inline std::vector<pten::DenseTensor*> SetKernelOutput(
    const std::vector<pten::DenseTensorMeta>& metas,
    Backend backend,
    std::vector<Tensor>* out) {
  size_t n = metas.size();
  out->reserve(n);
  std::vector<pten::DenseTensor*> results(n);
  for (size_t i = 0; i < n; ++i) {
    auto tensor_ptr = std::make_shared<pten::DenseTensor>(
        pten::make_intrusive<SharedStorage>(pten::TransToFluidPlace(backend)),
        metas[i]);
    results[i] = tensor_ptr.get();
    out->emplace_back();
    out->back().set_impl(tensor_ptr);
  }
  return results;
}

}  // namespace experimental
}  // namespace paddle
