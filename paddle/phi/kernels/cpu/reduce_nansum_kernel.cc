// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/reduce_nansum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void NansumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const IntArray& dims,
                  DataType out_dtype,
                  bool keep_dim,
                  DenseTensor* out) {
  if (out_dtype == DataType::UNDEFINED && out->dtype() != x.dtype()) {
    out_dtype = out->dtype();
  }

  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    if (out_dtype == DataType::INT64) {
      Full<int64_t, Context>(dev_ctx, out->dims(), 0, out);
    } else {
      Full<T, Context>(dev_ctx, out->dims(), 0, out);
    }
    return;
  }

  // Replace NaN with 0
  DenseTensor cleaned_x;
  cleaned_x.Resize(x.dims());
  dev_ctx.template Alloc<T>(&cleaned_x);
  const T* x_data = x.data<T>();
  T* clean_data = cleaned_x.data<T>();
  int64_t numel = x.numel();
  for (int64_t i = 0; i < numel; ++i) {
    clean_data[i] = (x_data[i] != x_data[i]) ? static_cast<T>(0) : x_data[i];
  }

  // Delegate to SumRawKernel
  bool reduce_all = recompute_reduce_all(x, dims);
  SumRawKernel<T, Context>(
      dev_ctx, cleaned_x, dims, keep_dim, reduce_all, out_dtype, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(nansum,
                   CPU,
                   ALL_LAYOUT,
                   phi::NansumKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int16_t,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
