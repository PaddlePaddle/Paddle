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

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
namespace phi {

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  dev_ctx.template Alloc<T>(out);
  if (x.dims() == y.dims()) {
    SameDimsElementwiseCompute<SameDimsMultiplyFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    DenseTensor x_fp32 = phi::Cast<T, Context>(dev_ctx, x, DataType::FLOAT32);
    DenseTensor y_fp32 = phi::Cast<T, Context>(dev_ctx, y, DataType::FLOAT32);
    DataType final_out_dtype = out->dtype();
    if (final_out_dtype == DataType::UNDEFINED) {
      final_out_dtype = x.dtype();
    }
    if constexpr (std::is_same_v<T, phi::dtype::float16> ||
                  std::is_same_v<T, phi::dtype::bfloat16>) {
      if (final_out_dtype == DataType::FLOAT32) {
        if (x_dims.size() >= y_dims.size()) {
          funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
              dev_ctx, x_fp32, y_fp32, funcs::MultiplyFunctor<T>(), out, -1);
        } else {
          funcs::ElementwiseCompute<funcs::InverseMultiplyFunctor<T>, T>(
              dev_ctx,
              x_fp32,
              y_fp32,
              funcs::InverseMultiplyFunctor<T>(),
              out,
              -1);
        }
      } else {
        DenseTensor intermediate_result;
        intermediate_result.set_meta(out->meta());
        if (x_dims.size() >= y_dims.size()) {
          funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
              dev_ctx,
              x_fp32,
              y_fp32,
              funcs::MultiplyFunctor<T>(),
              &intermediate_result,
              -1);
        } else {
          funcs::ElementwiseCompute<funcs::InverseMultiplyFunctor<T>, T>(
              dev_ctx,
              x_fp32,
              y_fp32,
              funcs::InverseMultiplyFunctor<T>(),
              &intermediate_result,
              -1);
        }

        phi::CastKernel<float, Context>(
            dev_ctx, intermediate_result, final_out_dtype, out);
      }
    } else {
      if (x_dims.size() >= y_dims.size()) {
        funcs::ElementwiseCompute<funcs::MultiplyFunctor<T>, T>(
            dev_ctx, x, y, funcs::MultiplyFunctor<T>(), out, -1);
      } else {
        funcs::ElementwiseCompute<funcs::InverseMultiplyFunctor<T>, T>(
            dev_ctx, x, y, funcs::InverseMultiplyFunctor<T>(), out, -1);
      }
    }
  }
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::phi::dtype::bfloat16;

PD_REGISTER_KERNEL(multiply,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
