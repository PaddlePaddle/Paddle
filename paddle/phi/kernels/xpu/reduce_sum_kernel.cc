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

#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/reduce.h"

namespace phi {

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out) {
  if (out_dtype == DataType::UNDEFINED && out->dtype() != x.dtype()) {
    out_dtype = out->dtype();
  }
  XPUReduce<Context, T, phi::SumFunctor>(
      dev_ctx, x, dims.GetData(), keep_dim, reduce_all, out_dtype, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_raw,
                   XPU,
                   ALL_LAYOUT,
                   phi::SumRawKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int8_t,
                   int,
                   int64_t,
                   bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
