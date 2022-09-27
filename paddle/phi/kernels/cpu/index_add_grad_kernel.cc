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

#include "paddle/phi/kernels/index_add_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"

namespace phi {

template <typename T, typename Context>
void IndexAddGradKernel(const Context& ctx,
                        const DenseTensor& index,
                        const DenseTensor& add_value,
                        const DenseTensor& out_grad,
                        int axis,
                        DenseTensor* x_grad,
                        DenseTensor* add_value_grad) {
  if (axis < 0) {
    axis += out_grad.dims().size();
  }
  const auto& index_type = index.dtype();

  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  // get x_grad: copy out_grad to x_grad.
  ctx.template Alloc<T>(x_grad);
  phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);

  auto inputs = out_grad;
  // get add_value_grad by using index_select(out_grad, index, axis)
  if (index_type == phi::DataType::INT32) {
    IndexSelectInner<Context, T, int>(
        ctx, &inputs, index, add_value_grad, axis);
  } else if (index_type == phi::DataType::INT64) {
    IndexSelectInner<Context, T, int64_t>(
        ctx, &inputs, index, add_value_grad, axis);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexAddGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
