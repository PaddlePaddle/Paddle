
// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/common/flags.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/prod_kernel.h"
#include "paddle/phi/kernels/reduce_all_kernel.h"
#include "paddle/phi/kernels/reduce_amax_kernel.h"
#include "paddle/phi/kernels/reduce_amin_kernel.h"
#include "paddle/phi/kernels/reduce_any_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/kernels/reduce_nansum_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);
COMMON_DECLARE_bool(force_stride_compute_contig_out);

namespace phi {

inline void PrepareStridedOut_reduce(DenseTensor* out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "should not be called!"));
  }
  auto meta = out->meta();
  meta.strides = meta.calc_strides(out->dims());
  out->set_meta(meta);
}

template <typename T, typename Context>
void AMaxStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::AMaxKernel<T, Context>(dev_ctx, x, dims, keep_dim, out);
}

template <typename T, typename Context>
void AMinStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::AMinKernel<T, Context>(dev_ctx, x, dims, keep_dim, out);
}

template <typename T, typename Context>
void MaxStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::MaxKernel<T, Context>(dev_ctx, x, dims, keep_dim, out);
}

template <typename T, typename Context>
void MinStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::MinKernel<T, Context>(dev_ctx, x, dims, keep_dim, out);
}

template <typename T, typename Context>
void ProdStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const IntArray& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::ProdKernel<T, Context>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void AllStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::AllKernel<T, Context>(dev_ctx, x, dims, keep_dim, out);
}

template <typename T, typename Context>
void AnyStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& dims,
                     bool keep_dim,
                     DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::AnyKernel<T, Context>(dev_ctx, x, dims, keep_dim, out);
}

template <typename T, typename Context>
void SumStrideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& dims,
                     DataType out_dtype,
                     bool keep_dim,
                     DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::SumKernel<T, Context>(dev_ctx, x, dims, out_dtype, keep_dim, out);
}

template <typename T, typename Context>
void NansumStrideKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& dims,
                        DataType out_dtype,
                        bool keep_dim,
                        DenseTensor* out) {
  PrepareStridedOut_reduce(out);
  phi::NansumKernel<T, Context>(dev_ctx, x, dims, out_dtype, keep_dim, out);
}

template <typename T, typename Context>
void MeanStrideKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const IntArray& dims,
                      bool keep_dim,
                      DenseTensor* out) {
  PrepareStridedOut_reduce(out);

  phi::MeanKernel<T, Context>(dev_ctx, x, dims, keep_dim, out);
}

}  // namespace phi

using float16 = phi::float16;
using bfloat16 = phi::bfloat16;
using complex64 = phi::complex64;
using complex128 = phi::complex128;

PD_REGISTER_KERNEL(
    amax, GPU, STRIDED, phi::AMaxStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    amin, GPU, STRIDED, phi::AMinStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    max, GPU, STRIDED, phi::MaxStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    min, GPU, STRIDED, phi::MinStrideKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(prod,
                   GPU,
                   STRIDED,
                   phi::ProdStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_REGISTER_KERNEL(any,
                   GPU,
                   STRIDED,
                   phi::AnyStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(all,
                   GPU,
                   STRIDED,
                   phi::AllStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(sum,
                   GPU,
                   STRIDED,
                   phi::SumStrideKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(nansum,
                   GPU,
                   STRIDED,
                   phi::NansumStrideKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(mean,
                   GPU,
                   STRIDED,
                   phi::MeanStrideKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::complex64,
                   phi::complex128) {}

#endif
