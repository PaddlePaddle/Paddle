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
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

PHI_DECLARE_bool(use_stride_kernel);

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

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

std::shared_ptr<phi::SparseCooTensor> TensorToSparseCooTensor(
    const Tensor& tensor) {
  return std::static_pointer_cast<phi::SparseCooTensor>(tensor.impl());
}
/* ----------------- for infer_meta --------------------- */

phi::MetaTensor MakeMetaTensor(const phi::TensorBase& tensor) {
  return phi::MetaTensor(tensor);
}

std::vector<phi::MetaTensor> MakeMetaTensor(
    const std::vector<const phi::TensorBase*>& tensors) {
  std::vector<phi::MetaTensor> meta_tensors;
  meta_tensors.reserve(tensors.size());
  for (const auto* t : tensors) {
    meta_tensors.emplace_back(*t);
  }
  return meta_tensors;
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
    const std::vector<const phi::SelectedRows*>& tensors) {
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

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::SparseCooTensor>& tensor) {
  if (tensor) {
    return {phi::MetaTensor(*tensor)};
  }
  return phi::MetaTensor();
}

phi::MetaTensor MakeMetaTensor(
    const paddle::optional<phi::SparseCsrTensor>& tensor) {
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
  if (!out) {
    return nullptr;
  }
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

phi::DenseTensor* ProcessStrideBackup(phi::DenseTensor** tensor) {
  if (!FLAGS_use_stride_kernel || *tensor == nullptr ||
      !(*tensor)->IsInitialized() || (*tensor)->meta().is_contiguous()) {
    return nullptr;
  } else {
    phi::DenseTensor* backup = *tensor;
    *tensor = new phi::DenseTensor();
    return backup;
  }
}

std::vector<phi::DenseTensor*> ProcessStrideBackup(
    std::vector<phi::DenseTensor*>* tensor) {
  std::vector<phi::DenseTensor*> backup;
  backup.reserve(tensor->size());
  for (auto& t : *tensor) {
    if (!FLAGS_use_stride_kernel || t == nullptr || !t->IsInitialized() ||
        t->meta().is_contiguous()) {
      backup.emplace_back(nullptr);
    } else {
      backup.emplace_back(t);
      t = new phi::DenseTensor();
    }
  }
  return backup;
}

phi::SelectedRows* ProcessStrideBackup(phi::SelectedRows** tensor) {
  return nullptr;
}

template <typename Context>
void TransStride(const Context& dev_ctx,
                 phi::DenseTensor* from,
                 phi::DenseTensor* to) {
  if (to) {
    PD_VISIT_ALL_TYPES(to->dtype(), "StridedCopyKernel", ([&] {
                         phi::StridedCopyKernel<data_t, Context>(
                             dev_ctx,
                             *from,
                             phi::vectorize<int64_t>(to->dims()),
                             phi::vectorize<int64_t>(to->strides()),
                             to->offset(),
                             to);
                       }));
    delete from;
  }
}

template <typename Context>
void TransStride(const Context& dev_ctx,
                 const std::vector<phi::DenseTensor*>& from,
                 const std::vector<phi::DenseTensor*>& to) {
  for (size_t i = 0; i < to.size(); i++) {
    if (to[i]) {
      PD_VISIT_ALL_TYPES(to[i]->dtype(), "StridedCopyKernel", ([&] {
                           phi::StridedCopyKernel<data_t, Context>(
                               dev_ctx,
                               *from[i],
                               phi::vectorize<int64_t>(to[i]->dims()),
                               phi::vectorize<int64_t>(to[i]->strides()),
                               to[i]->offset(),
                               to[i]);
                         }));
      delete from[i];
    }
  }
}

void TransStride(phi::DeviceContext* dev_ctx,
                 phi::DenseTensor* from,
                 phi::DenseTensor* to) {
  if (to) {
    auto* cpu_ctx = dynamic_cast<phi::CPUContext*>(dev_ctx);
    if (cpu_ctx) {
      PD_VISIT_ALL_TYPES(to->dtype(), "StridedCopyKernel", ([&] {
                           phi::StridedCopyKernel<data_t, phi::CPUContext>(
                               *cpu_ctx,
                               *from,
                               phi::vectorize<int64_t>(to->dims()),
                               phi::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      delete from;
      return;
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(dev_ctx);
    if (gpu_ctx) {
      PD_VISIT_ALL_TYPES(to->dtype(), "StridedCopyKernel", ([&] {
                           phi::StridedCopyKernel<data_t, phi::GPUContext>(
                               *gpu_ctx,
                               *from,
                               phi::vectorize<int64_t>(to->dims()),
                               phi::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      delete from;
      return;
    }
#endif
#ifdef PADDLE_WITH_XPU
    auto* xpu_ctx = dynamic_cast<phi::XPUContext*>(dev_ctx);
    if (xpu_ctx) {
      PD_VISIT_ALL_TYPES(to->dtype(), "StridedCopyKernel", ([&] {
                           phi::StridedCopyKernel<data_t, phi::XPUContext>(
                               *xpu_ctx,
                               *from,
                               phi::vectorize<int64_t>(to->dims()),
                               phi::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      delete from;
      return;
    }
#endif
  }
}

void TransStrideLegacy(phi::DeviceContext* dev_ctx,
                       phi::DenseTensor* from,
                       phi::DenseTensor* to) {
  if (to) {
    auto* cpu_ctx = dynamic_cast<phi::CPUContext*>(dev_ctx);
    if (cpu_ctx) {
      PD_VISIT_ALL_TYPES(to->dtype(), "StridedCopyKernel", ([&] {
                           phi::StridedCopyKernel<data_t, phi::CPUContext>(
                               *cpu_ctx,
                               *from,
                               phi::vectorize<int64_t>(to->dims()),
                               phi::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      return;
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(dev_ctx);
    if (gpu_ctx) {
      PD_VISIT_ALL_TYPES(to->dtype(), "StridedCopyKernel", ([&] {
                           phi::StridedCopyKernel<data_t, phi::GPUContext>(
                               *gpu_ctx,
                               *from,
                               phi::vectorize<int64_t>(to->dims()),
                               phi::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      return;
    }
#endif
#ifdef PADDLE_WITH_XPU
    auto* xpu_ctx = dynamic_cast<phi::XPUContext*>(dev_ctx);
    if (xpu_ctx) {
      PD_VISIT_ALL_TYPES(to->dtype(), "StridedCopyKernel", ([&] {
                           phi::StridedCopyKernel<data_t, phi::XPUContext>(
                               *xpu_ctx,
                               *from,
                               phi::vectorize<int64_t>(to->dims()),
                               phi::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      return;
    }
#endif
  }
}

void TransStride(phi::DeviceContext* dev_ctx,
                 const std::vector<phi::DenseTensor*>& from,
                 const std::vector<phi::DenseTensor*>& to) {
  for (size_t i = 0; i < to.size(); i++) {
    if (to[i]) {
      auto* cpu_ctx = dynamic_cast<phi::CPUContext*>(dev_ctx);
      if (cpu_ctx) {
        PD_VISIT_ALL_TYPES(to[i]->dtype(), "StridedCopyKernel", ([&] {
                             phi::StridedCopyKernel<data_t, phi::CPUContext>(
                                 *cpu_ctx,
                                 *from[i],
                                 phi::vectorize<int64_t>(to[i]->dims()),
                                 phi::vectorize<int64_t>(to[i]->strides()),
                                 to[i]->offset(),
                                 to[i]);
                           }));
        delete from[i];
        continue;
      }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      auto* gpu_ctx = dynamic_cast<phi::GPUContext*>(dev_ctx);
      if (gpu_ctx) {
        PD_VISIT_ALL_TYPES(to[i]->dtype(), "StridedCopyKernel", ([&] {
                             phi::StridedCopyKernel<data_t, phi::GPUContext>(
                                 *gpu_ctx,
                                 *from[i],
                                 phi::vectorize<int64_t>(to[i]->dims()),
                                 phi::vectorize<int64_t>(to[i]->strides()),
                                 to[i]->offset(),
                                 to[i]);
                           }));
        delete from[i];
        continue;
      }
#endif
#ifdef PADDLE_WITH_XPU
      auto* xpu_ctx = dynamic_cast<phi::XPUContext*>(dev_ctx);
      if (xpu_ctx) {
        PD_VISIT_ALL_TYPES(to[i]->dtype(), "StridedCopyKernel", ([&] {
                             phi::StridedCopyKernel<data_t, phi::XPUContext>(
                                 *xpu_ctx,
                                 *from[i],
                                 phi::vectorize<int64_t>(to[i]->dims()),
                                 phi::vectorize<int64_t>(to[i]->strides()),
                                 to[i]->offset(),
                                 to[i]);
                           }));
        delete from[i];
        continue;
      }
#endif
    }
  }
}

void TransStride(phi::DeviceContext* dev_ctx,
                 phi::SelectedRows* from,
                 phi::SelectedRows* to) {}

/* ------------------ for auto parallel ----------------------- */

phi::distributed::DistMetaTensor MakeDistMetaTensor(
    const phi::TensorBase& tensor) {
  return phi::distributed::DistMetaTensor(tensor);
}

phi::distributed::DistTensor* SetKernelDistOutput(
    Tensor* out, const phi::distributed::TensorDistAttr& dist_attr) {
  if (out) {
    // TODO(chenweihang): now all dist case are nullptr
    if (out->impl() == nullptr) {
      auto dist_t = std::make_shared<phi::distributed::DistTensor>(phi::DDim(),
                                                                   dist_attr);
      out->set_impl(dist_t);
    }
    return static_cast<phi::distributed::DistTensor*>(out->impl().get());
  }
  return nullptr;
}

std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    std::vector<Tensor*> out) {
  std::vector<phi::distributed::DistTensor*> result;
  for (auto tmp : out) {
    if (tmp) {
      // TODO(GhostScreaming): now all dist case are nullptr
      if (tmp->impl() == nullptr) {
        phi::DenseTensor dense_t;
        // TODO(GhostScreaming): polish code, dist_attr is null now
        phi::distributed::TensorDistAttr dist_attr;
        auto dist_t =
            std::make_shared<phi::distributed::DistTensor>(dense_t, dist_attr);
        tmp->set_impl(dist_t);
      }
      result.emplace_back(
          static_cast<phi::distributed::DistTensor*>(tmp->impl().get()));
    } else {
      result.emplace_back(nullptr);
    }
  }
  return result;
}

std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    size_t out_size, std::vector<Tensor>* out) {
  out->reserve(out_size);
  std::vector<phi::distributed::DistTensor*> results(out_size);
  for (size_t i = 0; i < out_size; ++i) {
    phi::DenseTensor dense_t;
    // TODO(GhostScreaming): polish code, dist_attr is null now
    phi::distributed::TensorDistAttr dist_attr;
    auto dist_t =
        std::make_shared<phi::distributed::DistTensor>(dense_t, dist_attr);
    results[i] = dist_t.get();
    out->emplace_back();
    out->back().set_impl(dist_t);
  }
  return results;
}

}  // namespace experimental
}  // namespace paddle
