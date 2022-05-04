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

#include "paddle/phi/kernels/index_fill_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/cpu/index_fill_impl.h"

namespace phi {

template <typename T, typename Context>
void IndexFillGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const DenseTensor& index,
                         int axis,
                         float fill_value,
                         DenseTensor* x_grad) {
  phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
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

  auto fill_val = static_cast<T>(0);
  if (index_type == phi::DataType::INT32) {
    IndexFillInner<Context, T, int>(dev_ctx, index, x_grad, axis, fill_val);
  } else if (index_type == phi::DataType::INT64) {
    IndexFillInner<Context, T, int64_t>(dev_ctx, index, x_grad, axis, fill_val);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexFillGradKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   int,
                   int64_t) {}
