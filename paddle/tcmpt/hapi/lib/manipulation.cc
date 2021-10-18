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

#include "paddle/tcmpt/hapi/include/manipulation.h"

#include <memory>

#include "glog/logging.h"
#include "paddle/tcmpt/api/include/core.h"
#include "paddle/tcmpt/hapi/lib/kernel_generate.h"
#include "paddle/tcmpt/infershape/unary.h"

namespace paddle {
namespace experimental {

Tensor flatten(const Tensor& x, int start_axis, int stop_axis) {
  // 1. Get kernel signature and kernel
  auto kernel_signature =
      ParseKernelNameAndKeyByArgs("flatten_contiguous_range", x);
  VLOG(1) << kernel_signature.first;
  VLOG(1) << kernel_signature.second;
  VLOG(1) << pt::KernelFactory::Instance();

  auto kernel = pt::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_signature.first, kernel_signature.second);
  VLOG(1) << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_signature.second.backend());
  auto kernel_context = pt::KernelContext(*dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pt::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);
  kernel_context.EmplaceBackAttr(start_axis);
  kernel_context.EmplaceBackAttr(stop_axis);

  // 4. InferShape
  // TODO(chenweihang): how to auto selected infershape?
  auto out_meta = FlattenInferShape(dense_x->meta(), start_axis, stop_axis);

  // 5. Prepare outputs
  Tensor out;
  // TODO(chenweihang): deal with multiple outputs
  auto dense_out =
      std::make_shared<pt::DenseTensor>(out_meta, pt::TensorStatus());
  kernel_context.EmplaceBackOutput(dense_out);
  out.set_impl(dense_out);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}
}  // namespace experimental
}  // namespace paddle
