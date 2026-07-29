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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/cascade_sum.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {
namespace {

template <typename T, typename Context>
void CascadeSum(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& axes,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  funcs::TorchCompatibleReduceSum<T>(x.data<T>(),
                                     common::vectorize(x.dims()),
                                     common::vectorize(x.strides()),
                                     axes,
                                     out->data<T>(),
                                     out->numel());
}

bool IsCascadeSumDtype(DataType dtype) {
  return dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 ||
         dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16 ||
         dtype == DataType::COMPLEX64 || dtype == DataType::COMPLEX128;
}

// Integral types are left to the generic path: torch reduces them with
// binary_kernel_reduce_vec and integer addition is associative, so any
// summation order already matches.
template <typename T, typename Context>
bool TryCascadeSum(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& dims,
                   bool reduce_all,
                   DataType out_dtype,
                   DenseTensor* out) {
  const DataType compute_dtype =
      out_dtype == DataType::UNDEFINED ? x.dtype() : out_dtype;
  if (!IsCascadeSumDtype(compute_dtype)) return false;

  // On CPU torch reduces in the output dtype, materializing `self.to(dtype)`
  // first, see make_reduction() in aten/src/ATen/native/ReduceOpsUtils.h.
  const bool need_cast = compute_dtype != x.dtype();
  if (need_cast && !x.meta().is_contiguous()) return false;

  const auto axes =
      funcs::NormalizeReduceAxes(x.dims(), dims.GetData(), reduce_all);

  // should_use_acc_buffer() in aten/src/ATen/native/ReduceOps.cpp: for an
  // fp16/bf16 result whose 2d loop reduces both dimensions, torch sums a
  // float32 copy of the *original* input and rounds only the result, so the
  // cast down to fp16/bf16 must not happen first. The same-dtype case is
  // handled inside TorchCompatibleReduceSum.
  if (need_cast &&
      (compute_dtype == DataType::FLOAT16 ||
       compute_dtype == DataType::BFLOAT16) &&
      funcs::NeedsFloatAccBuffer(
          common::vectorize(x.dims()), common::vectorize(x.strides()), axes)) {
    DenseTensor x_fp32 = x.dtype() == DataType::FLOAT32
                             ? x
                             : Cast<T, Context>(dev_ctx, x, DataType::FLOAT32);
    DenseTensor acc_out;
    acc_out.Resize(out->dims());
    dev_ctx.template Alloc<float>(&acc_out);
    funcs::TorchCompatibleReduceSum<float>(x_fp32.data<float>(),
                                           common::vectorize(x_fp32.dims()),
                                           common::vectorize(x_fp32.strides()),
                                           axes,
                                           acc_out.data<float>(),
                                           acc_out.numel());
    CastKernel<float, Context>(dev_ctx, acc_out, compute_dtype, out);
    return true;
  }

  DenseTensor casted;
  if (need_cast) {
    casted = Cast<T, Context>(dev_ctx, x, compute_dtype);
  }
  const DenseTensor& src = need_cast ? casted : x;

  switch (compute_dtype) {
    case DataType::FLOAT32:
      CascadeSum<float, Context>(dev_ctx, src, axes, out);
      return true;
    case DataType::FLOAT64:
      CascadeSum<double, Context>(dev_ctx, src, axes, out);
      return true;
    case DataType::FLOAT16:
      CascadeSum<float16, Context>(dev_ctx, src, axes, out);
      return true;
    case DataType::BFLOAT16:
      CascadeSum<bfloat16, Context>(dev_ctx, src, axes, out);
      return true;
    case DataType::COMPLEX64:
      CascadeSum<complex64, Context>(dev_ctx, src, axes, out);
      return true;
    case DataType::COMPLEX128:
      CascadeSum<complex128, Context>(dev_ctx, src, axes, out);
      return true;
    default:
      return false;
  }
}

}  // namespace

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
      Full<int64_t, Context>(dev_ctx, out->dims(), 0, out);
    } else {
      Full<T, Context>(dev_ctx, out->dims(), 0, out);
    }
    return;
  }
  if (FLAGS_use_accuracy_compatible_kernel &&
      TryCascadeSum<T, Context>(dev_ctx, x, dims, reduce_all, out_dtype, out)) {
    return;
  }
  if constexpr (std::is_same_v<T, float16> || std::is_same_v<T, bfloat16>) {
    DenseTensor x_fp32 = Cast<T, Context>(dev_ctx, x, DataType::FLOAT32);
    DataType final_out_dtype = out_dtype;
    if (final_out_dtype == DataType::UNDEFINED) {
      final_out_dtype = x.dtype();
    }
    if (final_out_dtype == DataType::FLOAT32) {
      Reduce<CPUContext, float, funcs::SumFunctor>(dev_ctx,
                                                   x_fp32,
                                                   reduce_all,
                                                   dims.GetData(),
                                                   keep_dim,
                                                   DataType::UNDEFINED,
                                                   out);
    } else {
      DenseTensor intermediate_result;
      intermediate_result.set_meta(out->meta());
      Reduce<CPUContext, float, funcs::SumFunctor>(dev_ctx,
                                                   x_fp32,
                                                   reduce_all,
                                                   dims.GetData(),
                                                   keep_dim,
                                                   DataType::UNDEFINED,
                                                   &intermediate_result);

      CastKernel<float, Context>(
          dev_ctx, intermediate_result, final_out_dtype, out);
    }
  } else {
    Reduce<CPUContext, T, funcs::SumFunctor>(
        dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::SumRawKernel,
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
