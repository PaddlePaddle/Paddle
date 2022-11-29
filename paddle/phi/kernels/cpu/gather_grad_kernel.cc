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

#include "paddle/phi/kernels/gather_grad_kernel.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/gather.h"
#include "paddle/phi/kernels/funcs/scatter.h"

namespace phi {

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& index,
                      const DenseTensor& out_grad,
                      const Scalar& axis,
                      bool overwrite,
                      DenseTensor* x_grad) {
  const auto& index_type = index.dtype();
  auto axis_v = axis.to<int>();

  if (axis_v != 0) {
    if (index_type == phi::DataType::INT32) {
      phi::funcs::GatherV2GradFunction<T, int32_t>(
          dev_ctx, &out_grad, &index, axis_v, x_grad);
    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::GatherV2GradFunction<T, int64_t>(
          dev_ctx, &out_grad, &index, axis_v, x_grad);
    }
    return;
  }

  dev_ctx.template Alloc<T>(x_grad);

  auto dxt = EigenVector<T>::Flatten(*x_grad);
  auto& place = *dev_ctx.eigen_device();
  dxt.device(place) = dxt.constant(static_cast<T>(0));
  if (x_grad->numel() == 0) return;

  if (index_type == phi::DataType::INT32) {
    if (overwrite) {
      phi::funcs::ScatterAssign<T, int32_t>(dev_ctx, out_grad, index, x_grad);
    } else {
      phi::funcs::ScatterAssignAdd<T, int32_t>(
          dev_ctx, out_grad, index, x_grad);
    }
  } else if (index_type == phi::DataType::INT64) {
    if (overwrite) {
      phi::funcs::ScatterAssign<T, int64_t>(dev_ctx, out_grad, index, x_grad);
    } else {
      phi::funcs::ScatterAssignAdd<T, int64_t>(
          dev_ctx, out_grad, index, x_grad);
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The data type of Input(Index) of gather_grad must be int32 or int64 "
        "on CPU."));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::GatherGradKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int64_t,
                   phi::dtype::bfloat16) {}
