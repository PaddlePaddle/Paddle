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

#include "paddle/pten/hapi/include/creation.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/core.h"
#include "paddle/pten/api/include/infershape.h"
#include "paddle/pten/hapi/lib/kernel_dispatch.h"

namespace paddle {
namespace experimental {

Tensor full_like(const Tensor& x,
                 const Scalar& value,
                 paddle::experimental::DataType dtype) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "fill_any_like", kernel_key);

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(*dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackAttr(value);

  // 4. InferShape
  auto out_meta = UnchangedInferShape(dense_x->meta());

  // 5. Prepare outputs
  Tensor out;
  // InferDataType
  if (dtype != pten::DataType::UNDEFINED) {
    out_meta.type = dtype;
  }
  auto dense_out =
      std::make_shared<pten::DenseTensor>(out_meta, pten::TensorStatus());
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

Tensor ones_like(const Tensor& x, DataType dtype) {
  return full_like(x, 1, dtype);
}

Tensor zeros_like(const Tensor& x, DataType dtype) {
  return full_like(x, 0, dtype);
}

}  // namespace experimental
}  // namespace paddle
