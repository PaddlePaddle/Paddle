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

#include "paddle/pten/api/include/creation.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/infershape.h"

namespace paddle {
namespace experimental {

PD_DLL_DECL Tensor full(const std::vector<int64_t>& shape,
                        const Scalar& value,
                        DataType dtype,
                        Backend backend,
                        DataLayout layout) {
  // 1. Get kernel signature and kernel
  pten::KernelKey kernel_key{backend, layout, dtype};
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "fill_constant.scalar", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  kernel_context.EmplaceBackAttr(value);

  // 4. InferShape
  auto out_meta = pten::FullInferShape(shape, dtype, layout);

  // 5. Prepare outputs
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  Tensor out;
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor full_like(const Tensor& x,
                             const Scalar& value,
                             DataType dtype,
                             Backend backend,
                             DataLayout layout) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();

  DataType kernel_data_type =
      dtype == DataType::UNDEFINED ? kernel_key.dtype() : dtype;
  Backend kernel_backend =
      backend == Backend::UNDEFINED ? kernel_key.backend() : backend;
  DataLayout kernel_layout =
      layout == DataLayout::UNDEFINED ? kernel_key.layout() : layout;

  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "fill_any_like", {kernel_backend, kernel_layout, kernel_data_type});

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackAttr(value);

  // 4. InferShape
  auto out_meta = FullLikeInferShape(dense_x->meta(), dtype, layout);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          pten::TransToFluidPlace(kernel_backend));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor ones_like(const Tensor& x,
                             DataType dtype,
                             Backend backend,
                             DataLayout layout) {
  return full_like(x, 1, dtype, backend, layout);
}

PD_DLL_DECL Tensor zeros_like(const Tensor& x,
                              DataType dtype,
                              Backend backend,
                              DataLayout layout) {
  return full_like(x, 0, dtype, backend, layout);
}

}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Creation);
