// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/cum_maxmin_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void CummaxGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out_grad,
                      int axis,
                      int dtype,
                      DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, x_grad, static_cast<T>(0));
  if (axis < 0) {
    axis = axis + x.dims().size();
  }
  auto indices_type = phi::TransToPhiDataType(dtype);
  if (indices_type == DataType::INT32) {
    phi::funcs::gpu_scatter_add_kernel<T, int32_t>(*x_grad, axis, indices, out_grad, dev_ctx);
  } else if (indices_type == DataType::INT64) {
    phi::funcs::gpu_scatter_add_kernel<T, int64_t>(*x_grad, axis, indices, out_grad, dev_ctx);
  }
}

template <typename T, typename Context>
void CumminGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out_grad,
                      int axis,
                      int dtype,
                      DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, x_grad, static_cast<T>(0));
  if (axis < 0) {
    axis = axis + x.dims().size();
  }
  auto indices_type = phi::TransToPhiDataType(dtype);
  if (indices_type == DataType::INT32) {
    phi::funcs::gpu_scatter_add_kernel<T, int32_t>(*x_grad, axis, indices, out_grad, dev_ctx);
  } else if (indices_type == DataType::INT64) {
    phi::funcs::gpu_scatter_add_kernel<T, int64_t>(*x_grad, axis, indices, out_grad, dev_ctx);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cummax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CummaxGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}

PD_REGISTER_KERNEL(cummin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CumminGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
