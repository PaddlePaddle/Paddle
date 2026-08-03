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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

// Without a STRIDED kernel the dispatcher materializes a contiguous copy of any
// non-contiguous input before calling the reduce (NeedTransform2Contiguous() in
// paddle/phi/api/lib/data_transform.cc), which both costs a full copy of the
// input and destroys the original dimension order. torch reduces the strided
// view in place, so its accumulation order can only be reproduced when the
// kernel sees the real strides.
//
// The Eigen based CPU reduce path cannot handle strides, so the view is only
// forwarded when the FLAGS_use_accuracy_compatible_kernel cascade path will
// consume it; otherwise this kernel does the materialization itself and behaves
// exactly like the previous dispatch.
namespace phi {
namespace {

void PrepareStridedOut(DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      FLAGS_use_stride_kernel,
      true,
      common::errors::Fatal("FLAGS_use_stride_kernel is closed. Strided kernel "
                            "should not be called!"));
  auto meta = out->meta();
  meta.strides = meta.calc_strides(out->dims());
  out->set_meta(meta);
}

bool IsCascadeDtype(DataType dtype) {
  return dtype == DataType::FLOAT32 || dtype == DataType::FLOAT64 ||
         dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16 ||
         dtype == DataType::COMPLEX64 || dtype == DataType::COMPLEX128;
}

template <typename T, typename Context>
const DenseTensor& ResolveInput(const Context& dev_ctx,
                                const DenseTensor& x,
                                bool strides_supported,
                                DenseTensor* buffer) {
  if (strides_supported || x.meta().is_contiguous()) {
    return x;
  }
  *buffer = Contiguous<T, Context>(dev_ctx, x);
  return *buffer;
}

}  // namespace

template <typename T, typename Context>
void SumStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     DataType out_dtype,
                     bool keep_dim,
                     DenseTensor* out) {
  PrepareStridedOut(out);
  // Mirror how SumRawKernel derives the reduction dtype, because only its
  // cascade path reads x.strides(); anything else ends up in the Eigen reduce
  // or in CastKernel, both of which read the input as contiguous memory. A
  // dtype promotion between cascade dtypes is fine: the cascade path converts
  // the input with CastPreservingLayout, which keeps the strides of a dense
  // view exactly like torch's `self.to(dtype)` does.
  DataType effective_dtype = out_dtype;
  if (effective_dtype == DataType::UNDEFINED && out->dtype() != x.dtype()) {
    effective_dtype = out->dtype();
  }
  const DataType compute_dtype =
      effective_dtype == DataType::UNDEFINED ? x.dtype() : effective_dtype;
  const bool strides_supported =
      FLAGS_use_accuracy_compatible_kernel && IsCascadeDtype(compute_dtype);
  DenseTensor buffer;
  const DenseTensor& src =
      ResolveInput<T, Context>(dev_ctx, x, strides_supported, &buffer);
  SumKernel<T, Context>(dev_ctx, src, dims, out_dtype, keep_dim, out);
}

template <typename T, typename Context>
void MeanStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const IntArray& dims,
                      bool keep_dim,
                      DenseTensor* out) {
  PrepareStridedOut(out);
  // MeanRawKernel's cascade path covers exactly the IsCascadeDtype set. fp16
  // and bf16 are included: it reduces them through a float32 accumulator built
  // with CastPreservingLayout, so the strides survive.
  const bool strides_supported =
      FLAGS_use_accuracy_compatible_kernel && IsCascadeDtype(x.dtype());
  DenseTensor buffer;
  const DenseTensor& src =
      ResolveInput<T, Context>(dev_ctx, x, strides_supported, &buffer);
  MeanKernel<T, Context>(dev_ctx, src, dims, keep_dim, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(sum,
                   CPU,
                   STRIDED,
                   phi::SumStrideKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(mean,
                   CPU,
                   STRIDED,
                   phi::MeanStrideKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
