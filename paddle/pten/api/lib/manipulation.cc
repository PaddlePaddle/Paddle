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

#include "paddle/pten/api/include/manipulation.h"

#include <memory>

#include "glog/logging.h"
#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/infermeta/unary.h"

PT_DECLARE_MODULE(ManipulationCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(ManipulationCUDA);
#endif

namespace paddle {
namespace experimental {

PD_DLL_DECL Tensor flatten(const Tensor& x, int start_axis, int stop_axis) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "flatten_contiguous_range", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackAttr(start_axis);
  kernel_context.EmplaceBackAttr(stop_axis);

  // 4. InferShape
  auto out_meta = FlattenInferShape(dense_x->meta(), start_axis, stop_axis);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor cast(const Tensor& x, DataType out_dtype) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "cast", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackAttr(out_dtype);
  kernel_context.EmplaceBackAttr(dense_x->meta().dtype);

  // 4. InferShape
  auto out_meta = CastInferMeta(dense_x->meta(), out_dtype);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

PD_DLL_DECL Tensor reshape(const Tensor& x, const std::vector<int64_t>& shape) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "reshape2", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackAttr(shape);

  // 4. InferShape
  auto out_meta = InferShapeFromVecValue(dense_x->meta(), shape);

  // 5. Prepare outputs
  Tensor out;
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Manipulation);
