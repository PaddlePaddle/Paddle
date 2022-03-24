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

#include "paddle/phi/kernels/scatter_nd_add_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/scatter.h"

namespace phi {

template <typename T, typename Context>
void ScatterNdAddKernel(const Context &ctx,
                        const DenseTensor &x,
                        const DenseTensor &index,
                        const DenseTensor &updates,
                        DenseTensor *out) {
  // In place output: Out = X
  Copy(ctx, x, ctx.GetPlace(), true, out);
  const auto &index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s] or [%s].",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  if (index_type == phi::DataType::INT32) {
    phi::funcs::ScatterNdAdd<T, int32_t>(ctx, updates, index, out);
  } else {
    phi::funcs::ScatterNdAdd<T, int64_t>(ctx, updates, index, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(scatter_nd_add,
                   CPU,
                   ALL_LAYOUT,
                   phi::ScatterNdAddKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   uint8_t) {}
