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

#include "paddle/tcmpt/api/include/dot.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/tcmpt/core/convert_utils.h"
#include "paddle/tcmpt/core/dense_tensor.h"
#include "paddle/tcmpt/core/kernel_context.h"
#include "paddle/tcmpt/core/kernel_generate.h"
#include "paddle/tcmpt/infershape/unary.h"

namespace pt {

Tensor dot(const Tensor& x, const Tensor& y) {
  // 1. Get kernel signature and kernel
  auto kernel_signature = ParseKernelNameAndKeyByArgs("dot", x);
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
  auto dense_y = std::dynamic_pointer_cast<DenseTensor>(y.impl());
  kernel_context.EmplaceBackInput(dense_y);
  // TODO(chenweihang): add transform impl

  // 4. InferShape
  // TODO(chenweihang): how to auto selected infershape?
  auto out_dims = DotInferShape(dense_x->dims());

  // 5. Prepare outputs
  pt::Tensor out;
  // TODO(chenweihang): deal with multiple outputs
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
