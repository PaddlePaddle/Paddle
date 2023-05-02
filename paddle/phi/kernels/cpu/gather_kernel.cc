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

#include "paddle/phi/kernels/gather_kernel.h"

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/gather.h"

namespace phi {

template <typename T, typename Context>
void GatherKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& index,
                  const Scalar& axis,
                  DenseTensor* out) {
  const auto& index_type = index.dtype();
  auto axis_v = axis.to<int>();
  if (axis_v != 0) {
    if (index_type == phi::DataType::INT32) {
      phi::funcs::GatherV2Function<T, int32_t>(
          dev_ctx, &x, &index, axis_v, out);
    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::GatherV2Function<T, int64_t>(
          dev_ctx, &x, &index, axis_v, out);
    }
    return;
  }

  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0) {
    return;
  }

  if (index_type == phi::DataType::INT32) {
    phi::funcs::CPUGather<T, int>(dev_ctx, x, index, out);
  } else if (index_type == phi::DataType::INT64) {
    phi::funcs::CPUGather<T, int64_t>(dev_ctx, x, index, out);
  } else {
    PADDLE_THROW(
        phi::errors::InvalidArgument("The data type of Input(Index) of gather "
                                     "must be int32 or int64 on CPU."));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(gather,
                   CPU,
                   ALL_LAYOUT,
                   phi::GatherKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int64_t,
                   phi::dtype::bfloat16) {}
