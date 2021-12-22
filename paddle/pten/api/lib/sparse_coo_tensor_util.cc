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

#include "paddle/pten/api/include/sparse_coo_tensor_utils.h"

#include "glog/logging.h"

#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/include/core.h"

PT_DECLARE_MODULE(SparseCooTensorUtilCPU);

// #if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// PT_DECLARE_MODULE(SparseCooTensorUtilCUDA);
// #endif

namespace paddle {
namespace experimental {

PADDLE_API Tensor to_sparse_coo(const Tensor& x, const int64_t sparse_dim) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "to_sparse_coo", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackAttr(sparse_dim);

  // 4. Prepare outputs
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(
          pten::TransToFluidPlace(kernel_key.backend()));

  Tensor out;
  auto sparse_out =
      std::make_shared<pten::SparseCooTensor>(allocator, dense_x->meta());
  kernel_context.EmplaceBackOutput(sparse_out);
  out.set_impl(sparse_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}
}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(SparseCooTensorUtil);
