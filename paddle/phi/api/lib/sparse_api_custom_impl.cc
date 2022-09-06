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

#include "paddle/phi/api/lib/sparse_api_custom_impl.h"

#include <memory>

#include "glog/logging.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/sparse/unary.h"

namespace paddle {
namespace experimental {
namespace sparse {

Tensor to_sparse_coo_impl(const Tensor& x, const int64_t sparse_dim) {
  if (x.is_sparse_coo_tensor()) {
    return x;
  }

  // 1. Get kernel signature and kernel
  std::string kernel_name = "dense_to_coo";
  if (x.is_sparse_csr_tensor()) {
    kernel_name = "csr_to_coo";
  }

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  const auto& kernel = kernel_result.kernel;

  VLOG(6) << "add API kernel key: " << kernel_key;
  VLOG(6) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = phi::KernelContext(dev_ctx);

  // 3. Auto data transform
  if (x.is_sparse_csr_tensor()) {
    auto input = std::dynamic_pointer_cast<phi::SparseCsrTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  } else {
    auto input = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
    kernel_context.EmplaceBackAttr(sparse_dim);
  }

  // 4. InferMeta
  auto indices_meta =
      phi::DenseTensorMeta(phi::DataType::INT64, {1}, phi::DataLayout::NCHW);
  auto elements_meta = phi::DenseTensorMeta(x.dtype(), {1}, x.layout());

  // 5. Prepare outputs
  // create empty SparseCooTensor
  phi::DenseTensor non_zero_indices(std::make_shared<phi::Allocation>(),
                                    std::move(indices_meta));
  phi::DenseTensor non_zero_elements(std::make_shared<phi::Allocation>(),
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

Tensor to_sparse_csr_impl(const Tensor& x) {
  if (x.is_sparse_csr_tensor()) {
    return x;
  }
  // 1. Get kernel signature and kernel
  std::string kernel_name = "dense_to_csr";
  if (x.is_sparse_coo_tensor()) {
    kernel_name = "coo_to_csr";
  }

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  const auto& kernel = kernel_result.kernel;

  VLOG(6) << "add API kernel key: " << kernel_key;
  VLOG(6) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = phi::KernelContext(dev_ctx);

  // 3. Auto data transform
  if (x.is_sparse_coo_tensor()) {
    auto input = std::dynamic_pointer_cast<phi::SparseCooTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  } else {
    auto input = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  }

  Tensor api_output;
  auto kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);
  phi::MetaTensor meta_out(kernel_out);
  phi::sparse::UnchangedInferMeta(MakeMetaTensor(*x.impl()), &meta_out);

  kernel_context.EmplaceBackOutput(
      static_cast<phi::SparseCsrTensor*>(kernel_out));

  kernel(&kernel_context);

  return api_output;
}

Tensor to_dense_impl(const Tensor& x) {
  if (x.is_dense_tensor()) {
    return x;
  }

  // 1. Get kernel signature and kernel
  std::string kernel_name = "coo_to_dense";
  if (x.is_sparse_csr_tensor()) {
    kernel_name = "csr_to_dense";
  }

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();

  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);
  const auto& kernel = kernel_result.kernel;

  VLOG(6) << "add API kernel key: " << kernel_key;
  VLOG(6) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = phi::KernelContext(dev_ctx);

  // 3. Auto data transform
  if (x.is_sparse_coo_tensor()) {
    auto input = std::dynamic_pointer_cast<phi::SparseCooTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  } else {
    auto input = std::dynamic_pointer_cast<phi::SparseCsrTensor>(x.impl());
    kernel_context.EmplaceBackInput(input.get());
  }

  Tensor api_output;
  auto kernel_out =
      SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);
  phi::MetaTensor meta_out(kernel_out);
  phi::sparse::UnchangedInferMeta(MakeMetaTensor(*x.impl()), &meta_out);

  kernel_context.EmplaceBackOutput(static_cast<phi::DenseTensor*>(kernel_out));

  kernel(&kernel_context);

  return api_output;
}

}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
