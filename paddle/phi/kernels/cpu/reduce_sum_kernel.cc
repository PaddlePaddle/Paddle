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
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/cascade_sum.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

namespace phi {
namespace {

// Reduces `x` in `Dst` precision. When Src and Dst differ the input is
// converted the way torch's `self.to(dtype)` does: a non-overlapping-and-dense
// view keeps its strides, so a transposed view is still reduced in the original
// dimension order instead of the flattened one.
template <typename Src, typename Dst, typename Context>
void CascadeSum(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& axes,
                DenseTensor* out) {
  dev_ctx.template Alloc<Dst>(out);
  const auto shape = common::vectorize(x.dims());
  const auto x_strides = common::vectorize(x.strides());
  if constexpr (std::is_same_v<Src, Dst>) {
    funcs::TorchCompatibleReduceSum<Dst>(
        x.data<Dst>(), shape, x_strides, axes, out->data<Dst>(), out->numel());
  } else {
    std::vector<Dst> buffer;
    std::vector<int64_t> strides;
    funcs::CastPreservingLayout<Src, Dst>(
        x.data<Src>(), shape, x_strides, &buffer, &strides);
    funcs::TorchCompatibleReduceSum<Dst>(
        buffer.data(), shape, strides, axes, out->data<Dst>(), out->numel());
  }
}

bool IsCascadeSumDtype(DataType dtype) {
  return dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 ||
         dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16 ||
         dtype == DataType::COMPLEX64 || dtype == DataType::COMPLEX128;
}

bool IsIntegralSumDtype(DataType dtype) {
  return dtype == DataType::UINT8 || dtype == DataType::INT8 ||
         dtype == DataType::INT16 || dtype == DataType::INT32 ||
         dtype == DataType::INT64;
}

// Integral types are left to the generic path: torch reduces them with
// binary_kernel_reduce_vec and integer addition is associative, so any
// summation order already matches.
template <typename T, typename Context>
bool TryCascadeSum(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DataType out_dtype,
                   DenseTensor* out) {
  const DataType compute_dtype =
      out_dtype == DataType::UNDEFINED ? x.dtype() : out_dtype;

  // Reducing into an integral dtype truncates every element *before* summing,
  // because torch reduces in the output dtype. The fp16/bf16 branch of
  // SumRawKernel instead sums in float32 and truncates only the total, which
  // changes the value: four 0.5 give 0 in torch and numpy, but 2 there.
  if (IsIntegralSumDtype(compute_dtype) && compute_dtype != x.dtype()) {
    DenseTensor contiguous_x;
    if (!x.meta().is_contiguous()) {
      contiguous_x = Contiguous<T, Context>(dev_ctx, x);
    }
    DenseTensor casted = Cast<T, Context>(
        dev_ctx, x.meta().is_contiguous() ? x : contiguous_x, compute_dtype);
    Reduce<CPUContext, T, funcs::SumFunctor>(dev_ctx,
                                             casted,
                                             reduce_all,
                                             dims.GetData(),
                                             keep_dim,
                                             DataType::UNDEFINED,
                                             out);
    return true;
  }

  if (!IsCascadeSumDtype(compute_dtype)) return false;

  // On CPU torch reduces in the output dtype, materializing `self.to(dtype)`
  // first, see make_reduction() in aten/src/ATen/native/ReduceOpsUtils.h.
  const bool need_cast = compute_dtype != x.dtype();
  const auto axes =
      funcs::NormalizeReduceAxes(x.dims(), dims.GetData(), reduce_all);

  // should_use_acc_buffer() in aten/src/ATen/native/ReduceOps.cpp: for an
  // fp16/bf16 result whose 2d loop reduces both dimensions, torch reduces
  // `self.to(float32)` instead and rounds only the result. The predicate runs
  // on the iterator built from `self.to(compute_dtype)`, which for a promotion
  // is the cast layout, and for a same-dtype reduction is x's own layout
  // because `to()` returns self there.
  if (compute_dtype == DataType::FLOAT16 ||
      compute_dtype == DataType::BFLOAT16) {
    const auto shape = common::vectorize(x.dims());
    const auto x_strides = common::vectorize(x.strides());
    const auto iter_strides =
        need_cast ? funcs::CastTargetStrides(shape, x_strides) : x_strides;
    if (funcs::NeedsFloatAccBuffer(shape, iter_strides, axes)) {
      DenseTensor acc_out;
      acc_out.Resize(out->dims());
      CascadeSum<T, float, Context>(dev_ctx, x, axes, &acc_out);
      CastKernel<float, Context>(dev_ctx, acc_out, compute_dtype, out);
      return true;
    }
  }

  PD_VISIT_FLOATING_AND_COMPLEX_AND_2_TYPES(
      DataType::FLOAT16, DataType::BFLOAT16, compute_dtype, "CascadeSum", ([&] {
        CascadeSum<T, data_t, Context>(dev_ctx, x, axes, out);
      }));
  return true;
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
  if (TryCascadeSum<T, Context>(
          dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out)) {
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
