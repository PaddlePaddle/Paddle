// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/take_along_axis_kernel.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/gather_scatter_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TakeAlongAxisKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& index,
                         int axis,
                         DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      paddle::platform::is_gpu_place(dev_ctx.GetPlace()),
      true,
      errors::PreconditionNotMet("This kernel only runs on GPU device."));

  out->Resize(index.dims());
  dev_ctx.template Alloc<T>(out);

  const auto& index_type =
      paddle::framework::TransToProtoVarType(index.dtype());
  if (index_type == paddle::framework::proto::VarType::INT32) {
    paddle::operators::gpu_gather_kernel<T, int32_t>(
        x, axis, index, *out, dev_ctx);
  } else if (index_type == paddle::framework::proto::VarType::INT64) {
    paddle::operators::gpu_gather_kernel<T, int64_t>(
        x, axis, index, *out, dev_ctx);
  }
}

}  // namespace  phi

PD_REGISTER_KERNEL(take_along_axis,
                   GPU,
                   ALL_LAYOUT,
                   phi::TakeAlongAxisKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16) {}
