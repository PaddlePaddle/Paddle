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

#include "paddle/pten/api/include/linalg.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/infershape.h"

namespace paddle {
namespace experimental {

PD_DLL_DECL Tensor dot(const Tensor& x, const Tensor& y) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "dot", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  // TODO(chenweihang): add transform impl

  // 4. InferShape
  auto out_meta = DotInferShape(dense_x->meta(), dense_y->meta());

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

PD_DLL_DECL Tensor matmul(const Tensor& x,
                          const Tensor& y,
                          bool transpose_x,
                          bool transpose_y) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "matmul_v2", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackInput(dense_y);
  kernel_context.EmplaceBackAttr(transpose_x);
  kernel_context.EmplaceBackAttr(transpose_y);
  // TODO(chenweihang): add transform impl

  // 4. InferShape
  auto out_meta = MatmulInferShape(
      dense_x->meta(), dense_y->meta(), transpose_x, transpose_y);

  // 5. Prepare outputs
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_out = std::make_shared<pten::DenseTensor>(allocator, out_meta);
  kernel_context.EmplaceBackOutput(dense_out);

  Tensor out;
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Linalg);
