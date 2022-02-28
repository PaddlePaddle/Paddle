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

#include "paddle/phi/api/include/sparse_api.h"

#include <memory>
#include "glog/logging.h"
#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"

PD_DECLARE_KERNEL(dense_to_sparse_coo, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_csr_to_coo, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(dense_to_sparse_csr, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_coo_to_csr, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_coo_to_dense, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_csr_to_dense, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(dense_to_sparse_coo, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_csr_to_coo, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(dense_to_sparse_csr, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_coo_to_csr, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_coo_to_dense, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(sparse_csr_to_dense, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace experimental {
namespace sparse {

PADDLE_API Tensor to_sparse_coo(const Tensor& x,
                                Backend backend,
                                const int64_t sparse_dim) {
  if (x.layout() == phi::DataLayout::SPARSE_COO) {
    return x;
  }
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  kernel_key_set.backend_set = kernel_key_set.backend_set | BackendSet(backend);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  std::string kernel_name = "dense_to_sparse_coo";
  if (x.layout() == phi::DataLayout::SPARSE_CSR) {
    kernel_name = "sparse_csr_to_coo";
  }

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);

  VLOG(6) << "to API kernel key: " << kernel_key;
  VLOG(6) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = phi::KernelContext(dev_ctx);

  // 3. Auto data transform
  if (x.layout() == phi::DataLayout::SPARSE_CSR) {
    auto input = std::dynamic_pointer_cast<phi::SparseCsrTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  } else {
    auto input = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
    kernel_context.EmplaceBackAttr(sparse_dim);
  }

  // 4. InferMeta
  auto indices_meta =
      phi::DenseTensorMeta(phi::DataType::INT64, {-1}, phi::DataLayout::NCHW);
  auto elements_meta = phi::DenseTensorMeta(x.dtype(), {-1}, x.layout());

  // 5. Prepare outputs
  // create empty SparseCooTensor
  phi::DenseTensor non_zero_indices(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPtenPlace(backend)),
      std::move(indices_meta));
  phi::DenseTensor non_zero_elements(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPtenPlace(backend)),
      std::move(elements_meta));
  auto coo = std::make_shared<phi::SparseCooTensor>(
      non_zero_indices, non_zero_elements, x.dims());

  kernel_context.EmplaceBackOutput(coo.get());
  Tensor out;
  out.set_impl(coo);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PADDLE_API Tensor to_sparse_csr(const Tensor& x, Backend backend) {
  if (x.layout() == phi::DataLayout::SPARSE_CSR) {
    return x;
  }
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  kernel_key_set.backend_set = kernel_key_set.backend_set | BackendSet(backend);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  std::string kernel_name = "dense_to_sparse_csr";
  if (x.layout() == phi::DataLayout::SPARSE_COO) {
    kernel_name = "sparse_coo_to_csr";
  }

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);

  VLOG(6) << "to API kernel key: " << kernel_key;
  VLOG(6) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = phi::KernelContext(dev_ctx);

  // 3. Auto data transform
  if (x.layout() == phi::DataLayout::SPARSE_COO) {
    auto input = std::dynamic_pointer_cast<phi::SparseCooTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  } else {
    auto input = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  }

  // 4. InferMeta
  auto crows_meta =
      phi::DenseTensorMeta(phi::DataType::INT64, {-1}, phi::DataLayout::NCHW);
  auto cols_meta =
      phi::DenseTensorMeta(phi::DataType::INT64, {-1}, phi::DataLayout::NCHW);
  auto elements_meta = phi::DenseTensorMeta(x.dtype(), {-1}, x.layout());

  // 5. Prepare outputs
  // create empty SparseCooTensor
  phi::DenseTensor non_zero_crows(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPtenPlace(backend)),
      std::move(crows_meta));
  phi::DenseTensor non_zero_cols(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPtenPlace(backend)),
      std::move(cols_meta));
  phi::DenseTensor non_zero_elements(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPtenPlace(backend)),
      std::move(elements_meta));
  auto csr = std::make_shared<phi::SparseCsrTensor>(
      non_zero_crows, non_zero_cols, non_zero_elements, x.dims());

  kernel_context.EmplaceBackOutput(csr.get());
  Tensor out;
  out.set_impl(csr);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PADDLE_API Tensor to_dense(const Tensor& x, Backend backend) {
  if (x.layout() != phi::DataLayout::SPARSE_CSR &&
      x.layout() != phi::DataLayout::SPARSE_COO) {
    return x;
  }
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  kernel_key_set.backend_set = kernel_key_set.backend_set | BackendSet(backend);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
  std::string kernel_name = "sparse_coo_to_dense";
  if (x.layout() == phi::DataLayout::SPARSE_CSR) {
    kernel_name = "sparse_csr_to_dense";
  }

  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);

  VLOG(6) << "to API kernel key: " << kernel_key;
  VLOG(6) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = phi::KernelContext(dev_ctx);

  // 3. Auto data transform
  if (x.layout() == phi::DataLayout::SPARSE_COO) {
    auto input = std::dynamic_pointer_cast<phi::SparseCooTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  } else {
    auto input = std::dynamic_pointer_cast<phi::SparseCsrTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  }

  // 4. InferMeta
  auto dense_meta = phi::DenseTensorMeta(x.dtype(), x.dims(), x.layout());

  // 5. Prepare outputs
  // create empty SparseCooTensor
  auto dense_out = std::make_shared<phi::DenseTensor>(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPtenPlace(backend)),
      std::move(dense_meta));

  kernel_context.EmplaceBackOutput(dense_out.get());
  Tensor out;
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

}  // namespace sparse
}  // namespace experimental
}  // namespace paddle

PD_REGISTER_API(SparseApi);
