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
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/memory/allocation/retry_allocator.h"
#include "paddle/phi/core/memory/malloc.h"
#include "paddle/phi/core/memory/mem_utils.h"
#include "paddle/phi/core/memory/stats.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/strided_copy_kernel.h"

PHI_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(enable_compact_mem);
COMMON_DECLARE_bool(use_virtual_memory_auto_growth);
COMMON_DECLARE_double(max_reserved_threshold_ratio);
COMMON_DECLARE_int64(max_reserved_threshold_in_gb);
COMMON_DECLARE_int64(cur_allocated_threshold_in_gb);
COMMON_DECLARE_bool(try_allocate);

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/kernel_factory.h"

namespace paddle::experimental {

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
                             common::vectorize<int64_t>(to->dims()),
                             common::vectorize<int64_t>(to->strides()),
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
                               common::vectorize<int64_t>(to[i]->dims()),
                               common::vectorize<int64_t>(to[i]->strides()),
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
                               common::vectorize<int64_t>(to->dims()),
                               common::vectorize<int64_t>(to->strides()),
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
                               common::vectorize<int64_t>(to->dims()),
                               common::vectorize<int64_t>(to->strides()),
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
                               common::vectorize<int64_t>(to->dims()),
                               common::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      delete from;
      return;
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto* custom_ctx = dynamic_cast<phi::CustomContext*>(dev_ctx);
    if (custom_ctx) {
      const phi::KernelKey& kernel_key = {phi::TransToPhiBackend(to->place()),
                                          phi::DataLayout::ALL_LAYOUT,
                                          to->dtype()};
      using kernel_signature = void (*)(const phi::DeviceContext&,
                                        const phi::DenseTensor&,
                                        const std::vector<int64_t>&,
                                        const std::vector<int64_t>&,
                                        int64_t,
                                        phi::DenseTensor*);
      PD_VISIT_KERNEL("strided_copy",
                      kernel_key,
                      kernel_signature,
                      false,
                      *custom_ctx,
                      *from,
                      common::vectorize<int64_t>(to->dims()),
                      common::vectorize<int64_t>(to->strides()),
                      to->offset(),
                      to);
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
                               common::vectorize<int64_t>(to->dims()),
                               common::vectorize<int64_t>(to->strides()),
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
                               common::vectorize<int64_t>(to->dims()),
                               common::vectorize<int64_t>(to->strides()),
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
                               common::vectorize<int64_t>(to->dims()),
                               common::vectorize<int64_t>(to->strides()),
                               to->offset(),
                               to);
                         }));
      return;
    }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto* custom_ctx = dynamic_cast<phi::CustomContext*>(dev_ctx);
    if (custom_ctx) {
      const phi::KernelKey& kernel_key = {phi::TransToPhiBackend(to->place()),
                                          phi::DataLayout::ALL_LAYOUT,
                                          to->dtype()};
      using kernel_signature = void (*)(const phi::DeviceContext&,
                                        const phi::DenseTensor&,
                                        const std::vector<int64_t>&,
                                        const std::vector<int64_t>&,
                                        int64_t,
                                        phi::DenseTensor*);
      PD_VISIT_KERNEL("strided_copy",
                      kernel_key,
                      kernel_signature,
                      false,
                      *custom_ctx,
                      *from,
                      common::vectorize<int64_t>(to->dims()),
                      common::vectorize<int64_t>(to->strides()),
                      to->offset(),
                      to);
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
                                 common::vectorize<int64_t>(to[i]->dims()),
                                 common::vectorize<int64_t>(to[i]->strides()),
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
                                 common::vectorize<int64_t>(to[i]->dims()),
                                 common::vectorize<int64_t>(to[i]->strides()),
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
                                 common::vectorize<int64_t>(to[i]->dims()),
                                 common::vectorize<int64_t>(to[i]->strides()),
                                 to[i]->offset(),
                                 to[i]);
                           }));
        delete from[i];
        continue;
      }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      auto* custom_ctx = dynamic_cast<phi::CustomContext*>(dev_ctx);
      if (custom_ctx) {
        const phi::KernelKey& kernel_key = {
            phi::TransToPhiBackend(to[i]->place()),
            phi::DataLayout::ALL_LAYOUT,
            to[i]->dtype()};
        using kernel_signature = void (*)(const phi::DeviceContext&,
                                          const phi::DenseTensor&,
                                          const std::vector<int64_t>&,
                                          const std::vector<int64_t>&,
                                          int64_t,
                                          phi::DenseTensor*);
        PD_VISIT_KERNEL("strided_copy",
                        kernel_key,
                        kernel_signature,
                        false,
                        *custom_ctx,
                        *from[i],
                        common::vectorize<int64_t>(to[i]->dims()),
                        common::vectorize<int64_t>(to[i]->strides()),
                        to[i]->offset(),
                        to[i]);
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

std::vector<phi::distributed::DistMetaTensor> MakeDistMetaTensor(
    const std::vector<const phi::TensorBase*>& tensors) {
  std::vector<phi::distributed::DistMetaTensor> meta_tensors;
  meta_tensors.reserve(tensors.size());
  for (const auto* t : tensors) {
    meta_tensors.emplace_back(*t);
  }
  return meta_tensors;
}

phi::distributed::DistTensor* SetKernelDistOutput(
    Tensor* out, const phi::distributed::ArgDistAttr& dist_attr) {
  PADDLE_ENFORCE_EQ(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr),
      true,
      common::errors::PreconditionNotMet(
          "Arg must be a single TensorDistAttr"));
  if (out) {
    if (out->impl() == nullptr) {
      auto dist_t = std::make_shared<phi::distributed::DistTensor>(
          phi::DDim(), paddle::get<0>(dist_attr));
      out->set_impl(dist_t);
    }
    return static_cast<phi::distributed::DistTensor*>(out->impl().get());
  }
  return nullptr;
}

std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    size_t out_size, std::vector<Tensor>* out) {
  std::vector<phi::distributed::DistTensor*> results(out_size);
  if (out->size() != out_size) {
    // Empty out vector
    out->reserve(out_size);
  }
  for (size_t i = 0; i < out_size; ++i) {
    if (out->size() != out_size) {
      auto dist_t = std::make_shared<phi::distributed::DistTensor>();
      out->emplace_back();
      out->back().set_impl(dist_t);
    }
    results[i] =
        static_cast<phi::distributed::DistTensor*>(out->at(i).impl().get());
  }
  return results;
}

std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    const phi::distributed::ArgDistAttr& dist_attr, std::vector<Tensor>* out) {
  PADDLE_ENFORCE_EQ(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          dist_attr),
      true,
      common::errors::PreconditionNotMet(
          "Arg must be a vector of TensorDistAttr"));
  const std::vector<phi::distributed::TensorDistAttr>& dist_attrs =
      PADDLE_GET_CONST(std::vector<phi::distributed::TensorDistAttr>,
                       dist_attr);
  auto out_size = dist_attrs.size();
  std::vector<phi::distributed::DistTensor*> results(out_size);
  // TODO(GhostScreaming): Inplace outputs are initialized, just set their
  // dist_attr.
  if (out->size() == out_size) {
    VLOG(3) << "Outputs are inplace vector Tensors, SKIP set dist_attr for out "
            << "to avoid changing the inplaced input";
    for (size_t i = 0; i < out_size; ++i) {
      results[i] =
          static_cast<phi::distributed::DistTensor*>(out->at(i).impl().get());
      continue;
      // auto t =
      //     static_cast<phi::distributed::DistTensor*>(out->at(i).impl().get());
      // auto dist_t = std::make_shared<phi::distributed::DistTensor>(
      //     t->shared_value(), t->dims(), dist_attrs[i]);
      // out->at(i) = Tensor();
      // out->at(i).set_impl(dist_t);
      // results[i] = dist_t.get();
    }
  } else {
    out->reserve(out_size);
    for (size_t i = 0; i < out_size; ++i) {
      auto dist_t = std::make_shared<phi::distributed::DistTensor>(
          phi::DDim(), dist_attrs[i]);
      results[i] = dist_t.get();
      out->emplace_back();
      out->back().set_impl(dist_t);
    }
  }
  return results;
}

// For backward
std::vector<phi::distributed::DistTensor*> SetKernelDistOutput(
    std::vector<Tensor*> out) {
  std::vector<phi::distributed::DistTensor*> result;
  for (auto tmp : out) {
    if (tmp) {
      // TODO(GhostScreaming): now all dist case are nullptr
      if (tmp->impl() == nullptr) {
        auto dist_t = std::make_shared<phi::distributed::DistTensor>();
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

std::shared_ptr<phi::distributed::DistTensor> CreateKernelDistOutput(
    Tensor* out,
    bool set_dist_output_as_tensor_impl,
    const phi::distributed::TensorDistAttr& dist_attr) {
  if (out) {
    auto dist_output =
        std::make_shared<phi::distributed::DistTensor>(phi::DDim(), dist_attr);
    if (set_dist_output_as_tensor_impl) {
      VLOG(3) << "CreateKernelDistOutput function set generated output "
                 "dist_tensor as Tensor's impl";
      if (out->is_dist_tensor()) {
        VLOG(3) << "out is DistTensor, set DistAttr:" << dist_attr
                << " to generated DistOutput.";
        dist_output->unsafe_set_dist_attr(dist_attr);
      }
      out->set_impl(dist_output);
    }
    return dist_output;
  }
  VLOG(4) << "CreateKernelDistOutput with NULL out";
  return nullptr;
}

std::shared_ptr<phi::distributed::DistTensor> CreateKernelDistOutput(
    Tensor* out,
    bool set_dist_output_as_tensor_impl,
    const phi::distributed::ArgDistAttr& dist_attr) {
  auto& tensor_dist_attr =
      PADDLE_GET_CONST(phi::distributed::TensorDistAttr, dist_attr);
  return CreateKernelDistOutput(
      out, set_dist_output_as_tensor_impl, tensor_dist_attr);
}

std::shared_ptr<phi::distributed::DistTensor> CreateKernelDistOutput(
    Tensor* out, const phi::distributed::ArgDistAttr& dist_attr) {
  auto& tensor_dist_attr =
      PADDLE_GET_CONST(phi::distributed::TensorDistAttr, dist_attr);
  return CreateKernelDistOutput(out, false, tensor_dist_attr);
}

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
CreateKernelDistOutput(std::vector<Tensor*> out,
                       bool set_dist_output_as_tensor_impl,
                       const phi::distributed::ArgDistAttr& dist_attr) {
  auto tensor_dist_attrs = PADDLE_GET_CONST(
      std::vector<phi::distributed::TensorDistAttr>, dist_attr);
  PADDLE_ENFORCE_EQ(
      out.size(),
      tensor_dist_attrs.size(),
      common::errors::PreconditionNotMet(
          "out.size() [%d] and tensor_dist_attrs.size() [%d] not match",
          out.size(),
          tensor_dist_attrs.size()));
  auto size = tensor_dist_attrs.size();
  std::vector<std::shared_ptr<phi::distributed::DistTensor>> results;
  results.reserve(size);
  for (size_t i = 0; i < size; i++) {
    results.emplace_back(CreateKernelDistOutput(
        out[i], set_dist_output_as_tensor_impl, tensor_dist_attrs[i]));
  }
  return results;
}

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
CreateKernelDistOutput(std::vector<Tensor*> out,
                       bool set_dist_output_as_tensor_impl) {
  auto size = out.size();
  std::vector<std::shared_ptr<phi::distributed::DistTensor>> results;
  results.reserve(size);
  for (size_t i = 0; i < size; i++) {
    results.emplace_back(
        CreateKernelDistOutput(out[i], set_dist_output_as_tensor_impl));
  }
  return results;
}

void SetReplicatedDistAttrForOutput(
    phi::distributed::DistTensor* out,
    const phi::distributed::ProcessMesh& process_mesh) {
  if (out) {
    if (out->dims().size() == -1 || out->dims().size() == 0) {
      if (out->local_dims().size() != -1 && out->local_dims().size() != 0) {
        out->unsafe_set_dims(out->local_dims());
        VLOG(3)
            << "DistTensor out has empty shape, use its local value's shape";
      }
    }
    // For inplace output, we also need to set replicated dist attr
    auto dist_attr =
        phi::distributed::TensorDistAttr(common::vectorize(out->dims()));
    dist_attr.set_process_mesh(process_mesh);
    out->unsafe_set_dist_attr(dist_attr);
  }
}

/* ------------------ for Allocator ----------------------- */

// Helper: Find the largest single tensor size from size vector
static size_t FindMaxSingleRequest(const std::vector<size_t>& size_vec) {
  size_t max_single_req = 0;
  for (const auto& s : size_vec) {
    max_single_req = std::max(max_single_req, s);
  }
  return max_single_req;
}

// Helper: Check memory capacity and determine if action (compact/offload) is needed
// Returns: pair<need_action, reason_string>
static std::pair<bool, std::string> CheckMemoryCapacity(
    size_t max_free_size,
    size_t large_N_free_size,
    size_t remaining_hbm,
    size_t max_single_req,
    size_t req_total_size) {
  // Check 1: Can the largest single tensor fit in a contiguous block?
  // It can fit in either pool's max_free_size or a new block from driver (remaining_hbm)
  if (max_single_req > std::max(max_free_size, remaining_hbm)) {
    std::ostringstream oss;
    oss << "no large enough contiguous block: max_single_req=" << max_single_req
        << " > max(max_free_size=" << max_free_size
        << ", remaining_hbm=" << remaining_hbm << ")";
    return {true, oss.str()};
  }

  // Check 2: Is total capacity (pool_free + remaining_hbm) sufficient?
  if (large_N_free_size + remaining_hbm < req_total_size) {
    std::ostringstream oss;
    oss << "total capacity insufficient: large_N_free_size=" << large_N_free_size
        << " + remaining_hbm=" << remaining_hbm
        << " < req_total_size=" << req_total_size;
    return {true, oss.str()};
  }

  return {false, ""};
}

// Cache for GpuAvailableMemToAlloc - thread local to avoid cross-thread issues
static thread_local size_t g_cached_remaining_hbm = 0;
static thread_local size_t g_cached_max_reserved = 0;
static thread_local int g_cache_call_count = 0;
constexpr int kCacheRefreshInterval = 100;

// Helper: Get remaining HBM with caching
static size_t GetRemainingHbmCached(size_t max_reserved, bool force_refresh = false) {
  // Determine if cache needs refresh:
  // 1. Force refresh requested
  // 2. First call (g_cached_max_reserved == 0)
  // 3. max_reserved changed significantly (>1GB difference, indicating memory growth)
  // 4. Periodic refresh (every kCacheRefreshInterval calls)
  bool need_refresh = force_refresh ||
      (g_cached_max_reserved == 0) ||
      (max_reserved > g_cached_max_reserved + (1ULL << 30)) ||
      (++g_cache_call_count >= kCacheRefreshInterval);

  if (need_refresh) {
    g_cached_remaining_hbm = phi::backends::gpu::GpuAvailableMemToAlloc();
    g_cached_max_reserved = max_reserved;
    g_cache_call_count = 0;
    VLOG(6) << "[Compact] remaining_hbm cache refreshed: "
            << (g_cached_remaining_hbm >> 20) << "MB";
  }
  return g_cached_remaining_hbm;
}

// Helper: Calculate tensor sizes from meta information
static std::pair<size_t, std::vector<size_t>> CalTensorSize(
    const std::vector<phi::MetaTensor*>& meta_tensors,
    const std::string& api) {
  size_t req_total_size = 0;
  std::vector<size_t> sizes;
  for (auto& meta_tensor : meta_tensors) {
    if (meta_tensor == nullptr) continue;
    auto numel = meta_tensor->numel();
    if (numel == 0) continue;
    // Use absolute value for negative numel (e.g., -1 indicates dynamic shape)
    if (numel < 0) {
      numel = std::abs(numel);
      VLOG(6) << "[Compact] numel < 0, using abs: " << numel << " in " << api;
    }
    size_t tensor_size = static_cast<size_t>(numel) * phi::SizeOf(meta_tensor->dtype());
    sizes.push_back(tensor_size);
    req_total_size += tensor_size;
  }
  return {req_total_size, sizes};
}

void CheckAndDoCompact(const std::vector<phi::MetaTensor*>& meta_tensors,
                       std::string api) {
  if (!FLAGS_enable_compact_mem || !FLAGS_use_virtual_memory_auto_growth)
    return;
#if defined(PADDLE_WITH_CUDA)
  // Get current CUDA device
  int current_device_id;
  cudaError_t err = cudaGetDevice(&current_device_id);
  if (UNLIKELY(err != cudaSuccess)) {
    if (err == cudaErrorInitializationError) {
      VLOG(10) << "[Compact] Skipping: CUDA Context not initialized "
               << "(possibly due to DataLoader fork). API: " << api;
      cudaGetLastError();  // Clear error
      return;
    }
    VLOG(6) << "[Compact] Skipping: cudaGetDevice failed with error " << err;
    return;
  }

  // Get memory statistics
  const auto& device_prop =
      phi::backends::gpu::GetDeviceProperties(current_device_id);
  const size_t total_mem = device_prop.totalGlobalMem;
  const auto max_reserved =
      paddle::memory::DeviceMemoryStatPeakValue("Reserved", current_device_id);
  const auto cur_allocated = paddle::memory::DeviceMemoryStatCurrentValue(
      "Allocated", current_device_id);
  const size_t pool_free = max_reserved > cur_allocated ?
      max_reserved - cur_allocated : 0;

  constexpr float kGB = 1 << 30;

  // Condition 1: Check max_reserved threshold (early return)
  // Use ratio-based threshold by default, fallback to GB-based if explicitly set
  size_t threshold;
  if (FLAGS_max_reserved_threshold_in_gb > 0) {
    // Use explicit GB threshold (backward compatibility)
    threshold = static_cast<size_t>(FLAGS_max_reserved_threshold_in_gb) << 30;
    VLOG(10) << "[Compact] Using GB-based threshold: " << FLAGS_max_reserved_threshold_in_gb << "GB";
  } else {
    // Use ratio-based threshold (adapts to different GPU sizes)
    threshold = static_cast<size_t>(total_mem * FLAGS_max_reserved_threshold_ratio);
    VLOG(10) << "[Compact] Using ratio-based threshold: " << (threshold / kGB)
             << "GB (" << (FLAGS_max_reserved_threshold_ratio * 100) << "% of " << (total_mem / kGB) << "GB)";
  }

  if (max_reserved < threshold) {
    VLOG(10) << "[Compact] Skip: max_reserved=" << (max_reserved / kGB)
             << "GB < threshold=" << (threshold / kGB) << "GB";
    return;
  }

  // Calculate tensor sizes
  const auto [req_total_size, size_vec] = CalTensorSize(meta_tensors, api);
  if (req_total_size == 0) {
    VLOG(10) << "[Compact] Skip: req_total_size=0";
    return;
  }

  VLOG(10) << "[Compact] API: " << api
           << ", device: " << current_device_id
           << ", total_mem: " << (total_mem / kGB) << "GB"
           << ", max_reserved: " << (max_reserved / kGB) << "GB"
           << ", cur_allocated: " << (cur_allocated / kGB) << "GB"
           << ", pool_free: " << (pool_free / kGB) << "GB"
           << ", req_total: " << (req_total_size / kGB) << "GB"
           << ", num_tensors: " << size_vec.size();

  // Pre-calculate max_single_req (size_vec is immutable)
  const size_t max_single_req = FindMaxSingleRequest(size_vec);

  // Variables to store values from NeedCompact for reuse
  size_t remaining_hbm_for_post_compact = 0;
  size_t max_free_before_compact = 0;
  bool need_offload_first = false;

  auto place = phi::GPUPlace(current_device_id);

  // Lambda: Determine if compact is needed
  auto NeedCompact = [&]() -> bool {
    const auto [max_free_size, large_N_free_size] =
        paddle::memory::VmmMaxFreeSize(place, meta_tensors.size());

    VLOG(10) << "[Compact] Pool status: max_free_size=" << (max_free_size / kGB) << "GB"
             << ", large_N_free_size=" << (large_N_free_size / kGB) << "GB"
             << ", req_total_size=" << (req_total_size / kGB) << "GB";

    // Condition 2: If largest contiguous block can satisfy request
    if (req_total_size < max_free_size) {
      VLOG(10) << "[Compact] Skip: req_total_size < max_free_size";
      return false;
    }

    // Condition 3: TryAllocBatch simulation (pure CPU, low cost)
    if (FLAGS_try_allocate) {
      auto alloc_succ = paddle::memory::TryAllocBatch(place, size_vec);
      VLOG(10) << "[Compact] TryAllocBatch result: " << (alloc_succ ? "success" : "failed");
      if (alloc_succ) return false;
    }

    // TryAllocBatch failed, check if driver can help
    size_t remaining_hbm = GetRemainingHbmCached(max_reserved);

    VLOG(10) << "[Compact] After TryAllocBatch failed: remaining_hbm=" << (remaining_hbm / kGB) << "GB"
             << ", max_single_req=" << (max_single_req / kGB) << "GB"
             << ", pool_free=" << (pool_free / kGB) << "GB";

    // Pre-estimate: after compact, pool_free becomes contiguous at tail
    // and is adjacent to remaining_hbm, so total available = pool_free + remaining_hbm
    bool compact_alone_sufficient = (pool_free + remaining_hbm >= req_total_size);

    if (!compact_alone_sufficient) {
      // Need offload first, then compact
      need_offload_first = true;
      VLOG(1) << "[Compact] API: " << api
              << " pre-estimate: pool_free(" << (pool_free / kGB) << "GB)"
              << " + remaining_hbm(" << (remaining_hbm / kGB) << "GB)"
              << " < req_total(" << (req_total_size / kGB) << "GB)"
              << ", will offload first";
    }

    remaining_hbm_for_post_compact = remaining_hbm;
    max_free_before_compact = max_free_size;
    return true;
  };

  // Main logic
  if (NeedCompact()) {
    // If pre-estimate shows compact alone is not enough, offload first
    if (need_offload_first) {
      VLOG(1) << "[Compact] API: " << api << " triggering offload before compact...";
      auto offloaded_size =
          paddle::memory::allocation::RunOOMCallback(place, req_total_size);
      if (offloaded_size > 0) {
        VLOG(1) << "[Compact] Offload completed: offloaded_size=" << (offloaded_size / kGB) << "GB";
      } else {
        VLOG(1) << "[Compact] Offload returned 0, no memory freed";
      }
    }

    size_t compacted_size = paddle::memory::Compact(place);

    // Get pool status after compact
    const auto [max_free_after, large_N_free_after] =
        paddle::memory::VmmMaxFreeSize(place, meta_tensors.size());

    VLOG(1) << "[Compact] API: " << api
            << ", max_reserved=" << (max_reserved / kGB) << "GB"
            << ", cur_allocated=" << (cur_allocated / kGB) << "GB"
            << ", pool_free=" << (pool_free / kGB) << "GB"
            << ", compacted=" << (compacted_size / kGB) << "GB"
            << ", max_free_block: " << (max_free_before_compact / kGB) << "GB -> " << (max_free_after / kGB) << "GB"
            << (need_offload_first ? " (with offload)" : "");
  }
#endif
}

}  // namespace paddle::experimental
