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

#include "paddle/phi/kernels/reduce_sum_kernel.h"

#include <set>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

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
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    // When out_dtype is DataType::UNDEFINED and input is int32 or bool,
    // result is int64, but FullKernel out_dtype parameter is not used, we need
    // to set int64 explicitly.
    if (out_dtype == DataType::INT64) {
      FullKernel<int64_t, Context>(
          dev_ctx,
          phi::IntArray(common::vectorize(out->dims())),
          0,
          out_dtype,  // not used
          out);
    } else {
      FullKernel<T, Context>(dev_ctx,
                             phi::IntArray(common::vectorize(out->dims())),
                             0,
                             out_dtype,  // not used
                             out);
    }
    return;
  }
  if constexpr (std::is_same_v<T, phi::dtype::float16> ||
                std::is_same_v<T, phi::dtype::bfloat16>) {
    DenseTensor x_fp32 = phi::Cast<T, Context>(dev_ctx, x, DataType::FLOAT32);
    DataType final_out_dtype = out_dtype;
    if (final_out_dtype == DataType::UNDEFINED) {
      final_out_dtype = x.dtype();
    }
    if (final_out_dtype == DataType::FLOAT32) {
      phi::Reduce<CPUContext, float, phi::funcs::SumFunctor>(
          dev_ctx,
          x_fp32,
          reduce_all,
          dims.GetData(),
          keep_dim,
          phi::DataType::UNDEFINED,
          out);
    } else {
      DenseTensor intermediate_result;
      intermediate_result.set_meta(out->meta());
      phi::Reduce<CPUContext, float, phi::funcs::SumFunctor>(
          dev_ctx,
          x_fp32,
          reduce_all,
          dims.GetData(),
          keep_dim,
          phi::DataType::UNDEFINED,
          &intermediate_result);

      phi::CastKernel<float, Context>(
          dev_ctx, intermediate_result, final_out_dtype, out);
    }
  } else {
    phi::Reduce<CPUContext, T, phi::funcs::SumFunctor>(
        dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
  }
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(sum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::SumRawKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
