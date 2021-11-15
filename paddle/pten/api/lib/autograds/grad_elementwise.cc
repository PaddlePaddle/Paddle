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

#include "paddle/pten/api/include/grad_elementwise.h"

#include <memory>

#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/include/core.h"

PT_DECLARE_MODULE(GradElementwiseCPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_MODULE(GradElementwiseCUDA);
#endif

namespace paddle {
namespace experimental {

std::vector<Tensor> grad_elementwise_add(const Tensor& x,
                                         const Tensor& y,
                                         const Tensor& grad_out,
                                         int axis) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(grad_out);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "grad_elementwise_add", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  auto dense_grad_out =
      std::dynamic_pointer_cast<pten::DenseTensor>(grad_out.impl());
  kernel_context.EmplaceBackInput(dense_grad_out);

  kernel_context.EmplaceBackAttr(axis);
  // TODO(chenweihang): add transform impl

  // 5. Prepare outputs
  Tensor grad_x;
  Tensor grad_y;

  // TODO(chenweihang): deal with multiple outputs
  const auto allocator = std::make_shared<DefaultAllocator>(
      pten::TransToFluidPlace(kernel_key.backend()));
  auto dense_grad_x =
      std::make_shared<pten::DenseTensor>(allocator, dense_x->meta());
  auto dense_grad_y =
      std::make_shared<pten::DenseTensor>(allocator, dense_y->meta());
  kernel_context.EmplaceBackOutput(dense_grad_x);
  kernel_context.EmplaceBackOutput(dense_grad_y);
  grad_x.set_impl(dense_grad_x);
  grad_y.set_impl(dense_grad_y);

  // 6. Call kernel
  kernel(&kernel_context);

  return {grad_x, grad_y};
}

}  // namespace experimental
}  // namespace paddle
