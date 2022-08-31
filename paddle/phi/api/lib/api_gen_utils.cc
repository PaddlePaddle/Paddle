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

#include "paddle/phi/api/lib/api_gen_utils.h"

namespace paddle {
namespace experimental {

/* ------------------ for input ----------------------- */

std::shared_ptr<phi::DenseTensor> TensorToDenseTensor(const Tensor& tensor) {
  return std::static_pointer_cast<phi::DenseTensor>(tensor.impl());
}

paddle::optional<phi::DenseTensor> TensorToDenseTensor(
    const paddle::optional<Tensor>& tensor) {
  if (tensor) {
    return {*std::static_pointer_cast<phi::DenseTensor>(tensor->impl())};
  }
  return nullptr;
}

std::unique_ptr<std::vector<phi::DenseTensor*>> TensorToDenseTensor(
    const std::vector<Tensor>& tensors) {
  auto pt_tensors = std::make_unique<std::vector<phi::DenseTensor*>>();
  pt_tensors->reserve(tensors.size());

  for (const auto& t : tensors) {
    pt_tensors->push_back(
        std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()).get());
  }

  return pt_tensors;
}

std::vector<const phi::DenseTensor*> TensorToConstDenseTensorPtr(
    const std::vector<Tensor>& tensors) {
  std::vector<const phi::DenseTensor*> pt_tensors(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    pt_tensors[i] = static_cast<phi::DenseTensor*>(tensors[i].impl().get());
  }

  return pt_tensors;
}

paddle::optional<std::vector<const phi::DenseTensor*>>
TensorToConstDenseTensorPtr(
    const paddle::optional<std::vector<Tensor>>& tensors) {
  paddle::optional<std::vector<const phi::DenseTensor*>> pt_tensors;

  if (tensors) {
    pt_tensors =
        paddle::optional<std::vector<const phi::DenseTensor*>>(tensors->size());
    for (size_t i = 0; i < tensors->size(); ++i) {
      pt_tensors->at(i) =
          static_cast<phi::DenseTensor*>(tensors->at(i).impl().get());
    }
  }

  return pt_tensors;
}

std::shared_ptr<phi::SelectedRows> TensorToSelectedRows(const Tensor& tensor) {
  return std::static_pointer_cast<phi::SelectedRows>(tensor.impl());
}

paddle::optional<phi::SelectedRows> TensorToSelectedRows(
    const paddle::optional<Tensor>& tensor) {
  if (tensor) {
    return {*std::static_pointer_cast<phi::SelectedRows>(tensor->impl())};
  }
  return nullptr;
}

std::shared_ptr<phi::StringTensor> TensorToStringTensor(const Tensor& tensor) {
  return std::dynamic_pointer_cast<phi::StringTensor>(tensor.impl());
}

/* ----------------- for infer_meta --------------------- */

phi::MetaTensor MakeMetaTensor(const phi::TensorBase& tensor) {
  return phi::MetaTensor(tensor);
}

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::DenseTensor>& tensor) {
  if (tensor) {
    return {phi::MetaTensor(*tensor)};
  }
  return phi::MetaTensor();
}

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<const phi::DenseTensor*>& tensors) {
  std::vector<phi::MetaTensor> meta_tensors;
  meta_tensors.reserve(tensors.size());
  for (const auto* t : tensors) {
    meta_tensors.emplace_back(*t);
  }
  return meta_tensors;
}

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<phi::DenseTensor*>& tensors) {
  std::vector<phi::MetaTensor> meta_tensors;
  meta_tensors.reserve(tensors.size());
  for (auto* t : tensors) {
    meta_tensors.emplace_back(*t);
  }
  return meta_tensors;
}

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::SelectedRows>& tensor) {
  if (tensor) {
    return {phi::MetaTensor(*tensor)};
  }
  return phi::MetaTensor();
}

std::vector<phi::MetaTensor> MakeMetaTensor(
    const paddle::optional<std::vector<const phi::DenseTensor*>>& tensors) {
  std::vector<phi::MetaTensor> meta_tensors;
  if (tensors) {
    meta_tensors.reserve(tensors->size());
    for (auto* t : tensors.get()) {
      meta_tensors.emplace_back(*t);
    }
  }
  return meta_tensors;
}

/* ------------------ for output ----------------------- */

phi::DenseTensor* SetKernelOutput(Tensor* out) {
  if (out) {
    if (out->impl() == nullptr) {
      out->set_impl(std::make_shared<phi::DenseTensor>());
    }
    return static_cast<phi::DenseTensor*>(out->impl().get());
  }
  return nullptr;
}

std::vector<phi::DenseTensor*> SetKernelOutput(size_t out_size,
                                               std::vector<Tensor>* out) {
  out->reserve(out_size);
  std::vector<phi::DenseTensor*> results(out_size);
  for (size_t i = 0; i < out_size; ++i) {
    auto tensor_ptr = std::make_shared<phi::DenseTensor>();
    results[i] = tensor_ptr.get();
    out->emplace_back();
    out->back().set_impl(tensor_ptr);
  }
  return results;
}

std::vector<phi::DenseTensor*> SetInplaceVectorKernelOutput(
    size_t out_size, std::vector<Tensor>* out) {
  std::vector<phi::DenseTensor*> results(out->size(), nullptr);
  for (size_t i = 0; i < out->size(); ++i) {
    results[i] = static_cast<phi::DenseTensor*>(out->at(i).impl().get());
  }
  return results;
}

std::vector<phi::DenseTensor*> SetInplaceOptionalVectorKernelOutput(
    size_t out_size, const paddle::optional<std::vector<Tensor>>& out) {
  std::vector<phi::DenseTensor*> results;
  if (out) {
    results = std::vector<phi::DenseTensor*>(out->size(), nullptr);
    for (size_t i = 0; i < out->size(); ++i) {
      results[i] = static_cast<phi::DenseTensor*>(out->at(i).impl().get());
    }
  }
  return results;
}

std::vector<phi::DenseTensor*> SetKernelOutput(std::vector<Tensor*>* out) {
  std::vector<phi::DenseTensor*> results(out->size(), nullptr);
  for (size_t i = 0; i < out->size(); ++i) {
    if (out->at(i)) {
      auto tensor_ptr = std::make_shared<phi::DenseTensor>();
      results[i] = tensor_ptr.get();
      (*out)[i]->set_impl(tensor_ptr);
    }
  }
  return results;
}

phi::SelectedRows* SetSelectedRowsKernelOutput(Tensor* out) {
  if (!out->initialized()) {
    auto select_rows = std::make_shared<phi::SelectedRows>();
    out->set_impl(select_rows);
    return select_rows.get();
  }
  return static_cast<phi::SelectedRows*>(out->impl().get());
}

phi::TensorBase* SetSparseKernelOutput(Tensor* out, TensorType type) {
  if (!out->initialized()) {
    if (type == TensorType::SPARSE_COO) {
      auto sparse_tensor = std::make_shared<phi::SparseCooTensor>(
          phi::DenseTensor(), phi::DenseTensor(), phi::DDim{-1});
      out->set_impl(sparse_tensor);
      return sparse_tensor.get();
    } else if (type == TensorType::SPARSE_CSR) {
      auto sparse_tensor =
          std::make_shared<phi::SparseCsrTensor>(phi::DenseTensor(),
                                                 phi::DenseTensor(),
                                                 phi::DenseTensor(),
                                                 phi::DDim{-1, -1});
      out->set_impl(sparse_tensor);
      return sparse_tensor.get();
    } else {
      auto dense_tensor = std::make_shared<phi::DenseTensor>();
      out->set_impl(dense_tensor);
      return dense_tensor.get();
    }
  }
  return out->impl().get();
}

phi::TensorBase* SetStringsKernelOutput(Tensor* out, TensorType type) {
  if (!out->initialized()) {
    if (type == TensorType::STRING_TENSOR) {
      if (out->impl() == nullptr) {
        auto strings_tensor = std::make_shared<phi::StringTensor>();
        out->set_impl(strings_tensor);
      }
      return out->impl().get();
    }
  }
  return out->impl().get();
}

}  // namespace experimental
}  // namespace paddle
