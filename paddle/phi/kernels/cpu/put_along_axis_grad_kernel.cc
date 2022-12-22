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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/gather_scatter_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

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
  PADDLE_ENFORCE_EQ(
      paddle::platform::is_cpu_place(dev_ctx.GetPlace()),
      true,
      errors::PreconditionNotMet("PutAlongAxisGradOpKernel only runs on CPU."));

  const auto& index_type =
      paddle::framework::TransToProtoVarType(index.dtype());
  if (x_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    if (index_type == paddle::framework::proto::VarType::INT32) {
      paddle::operators::cpu_scatter_input_grad_kernel<T, int32_t>(
          // Here passing an unused argument out_grad, because it's
          // convenient to instantiate a bunch of template function with the
          // same arguments list.
          out_grad,
          axis,
          index,
          *x_grad,
          dev_ctx);
    } else {
      paddle::operators::cpu_scatter_input_grad_kernel<T, int64_t>(
          out_grad, axis, index, *x_grad, dev_ctx);
    }
  }

  if (value_grad) {
    value_grad->Resize(index.dims());
    value_grad->mutable_data<T>(dev_ctx.GetPlace());
    if (index_type == paddle::framework::proto::VarType::INT32) {
      paddle::operators::cpu_gather_kernel<T, int32_t>(
          out_grad, axis, index, *value_grad, dev_ctx);
    } else if (index_type == paddle::framework::proto::VarType::INT64) {
      paddle::operators::cpu_gather_kernel<T, int64_t>(
          out_grad, axis, index, *value_grad, dev_ctx);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisGradKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int64_t) {}
