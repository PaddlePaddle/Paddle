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

#include "paddle/phi/kernels/put_along_axis_kernel.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/gather_scatter_kernel.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"

namespace phi {

template <typename T, typename Context>
void PutAlongAxisKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& index,
                        const DenseTensor& value,
                        int axis,
                        const std::string& reduce,
                        DenseTensor* out) {
  PADDLE_ENFORCE_EQ(paddle::platform::is_gpu_place(dev_ctx.GetPlace()),
                    true,
                    errors::PreconditionNotMet(
                        "PutAlongAxisCUDAKernel only runs on GPU device."));

  const auto& index_type =
      paddle::framework::TransToProtoVarType(index.dtype());

  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  if (reduce == "add") {
    if (index_type == paddle::framework::proto::VarType::INT32) {
      paddle::operators::gpu_scatter_add_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == paddle::framework::proto::VarType::INT64) {
      paddle::operators::gpu_scatter_add_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "multiply" || reduce == "mul") {
    if (index_type == paddle::framework::proto::VarType::INT32) {
      paddle::operators::gpu_scatter_mul_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == paddle::framework::proto::VarType::INT64) {
      paddle::operators::gpu_scatter_mul_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else if (reduce == "assign") {
    if (index_type == paddle::framework::proto::VarType::INT32) {
      paddle::operators::gpu_scatter_assign_kernel<T, int32_t>(
          *out, axis, index, value, dev_ctx);
    } else if (index_type == paddle::framework::proto::VarType::INT64) {
      paddle::operators::gpu_scatter_assign_kernel<T, int64_t>(
          *out, axis, index, value, dev_ctx);
    }
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "can not support reduce: '%s' for scatter kernel, only "
        "support reduce op: 'add', 'assign', 'mul' and 'multiply', the "
        "defalut reduce op is 'assign' ",
        reduce));
    return;
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis,
                   GPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16) {}
