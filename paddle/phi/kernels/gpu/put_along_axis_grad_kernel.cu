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

#include "paddle/phi/kernels/put_along_axis_grad_kernel.h"

#include "paddle/fluid/operators/gather_scatter_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

template <typename T, typename Context>
void PutAlongAxisGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& index,
                            const DenseTensor& out_grad,
                            int axis,
                            const std::string& reduce,
                            DenseTensor* x_grad,
                            DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(paddle::platform::is_gpu_place(dev_ctx.GetPlace()),
                    true,
                    errors::PreconditionNotMet(
                        "PutAlongAxisGradOpCUDAKernel only runs on GPU."));

  const auto& index_type = index.dtype();
  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    if (index_type == DataType::INT32) {
      paddle::operators::gpu_scatter_input_grad_kernel<T, int32_t>(
          out_grad, axis, index, *x_grad, dev_ctx);
    } else {
      paddle::operators::gpu_scatter_input_grad_kernel<T, int64_t>(
          out_grad, axis, index, *x_grad, dev_ctx);
    }
  }
  if (value_grad) {
    value_grad->Resize(index.dims());
    dev_ctx.template Alloc<T>(value_grad);
    if (index_type == DataType::INT32) {
      paddle::operators::gpu_gather_kernel<T, int32_t>(
          out_grad,
          axis,
          index,
          *value_grad,
          dev_ctx);  // the gradient of scatter is gather
    } else if (index_type == DataType::INT64) {
      paddle::operators::gpu_gather_kernel<T, int64_t>(
          out_grad, axis, index, *value_grad, dev_ctx);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16) {}
