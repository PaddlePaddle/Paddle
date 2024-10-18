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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"
#include "paddle/phi/kernels/gather_kernel.h"

namespace phi {

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& index,
                      const DenseTensor& out_grad,
                      const Scalar& axis,
                      DenseTensor* x_grad) {
  const auto& index_type = index.dtype();
  auto axis_v = axis.to<int>();
  if (axis_v < 0) {
    axis_v += static_cast<int>(x.dims().size());
  }

  if (axis_v != 0) {
    if (index_type == DataType::INT32) {
      phi::funcs::GatherV2GradCUDAFunction<T, int32_t>(
          &out_grad, &index, axis_v, x_grad, dev_ctx);
    } else if (index_type == DataType::INT64) {
      phi::funcs::GatherV2GradCUDAFunction<T, int64_t>(
          &out_grad, &index, axis_v, x_grad, dev_ctx);
    }
    return;
  }

  dev_ctx.template Alloc<T>(x_grad);
  auto dxt = EigenVector<T>::Flatten(*x_grad);
  auto& place = *dev_ctx.eigen_device();
  dxt.device(place) = dxt.constant(static_cast<T>(0));
  if (out_grad.numel() == 0) return;
  if (index_type == DataType::INT32) {
    phi::funcs::GPUScatterAssign<T, int>(
        dev_ctx, out_grad, index, x_grad, false);
  } else if (index_type == DataType::INT64) {
    phi::funcs::GPUScatterAssign<T, int64_t>(
        dev_ctx, out_grad, index, x_grad, false);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The data type of Input(Index) of gather_grad must be int32 or int64 "
        "on GPU."));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gather_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::GatherGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
