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

#include "paddle/tcmpt/api/include/creation.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/tcmpt/api/include/dev/core.h"
#include "paddle/tcmpt/api/include/dev/creation.h"
#include "paddle/tcmpt/api/include/dev/infershape.h"
#include "paddle/tcmpt/core/kernel_generate.h"

namespace pt {

// template<class T>
// Tensor full_like(const Tensor& x, T value) {
Tensor full_like(const Tensor& x, float value) {
  // 1. Get kernel signature and kernel
  auto kernel_signature = ParseKernelNameAndKeyByArgs("fill_any_like", x);
  VLOG(1) << kernel_signature.first;
  VLOG(1) << kernel_signature.second;
  VLOG(1) << KernelFactory::Instance();

  auto kernel = KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_signature.first, kernel_signature.second);
  VLOG(1) << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_signature.second.backend());
  auto kernel_context = KernelContext(*dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);

  kernel_context.EmplaceBackAttr(value);

  // 4. InferShape
  auto out_dims = UnchangedInferShape(dense_x->dims());

  // 5. Prepare outputs
  pt::Tensor out;
  auto out_def = kernel.args_def().output_defs()[0];
  auto dense_out = std::make_shared<DenseTensor>(
      TensorMeta(out_dims, out_def.backend, out_def.dtype, out_def.layout),
      TensorStatus());
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

}  // namespace pt
