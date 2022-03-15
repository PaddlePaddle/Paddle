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

#include "paddle/phi/kernels/reduce_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               DataType out_dtype,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = false;
  SumRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                DenseTensor* out) {
  bool reduce_all = false;
  MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                DenseTensor* out) {
  bool reduce_all = false;
  ProdRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = false;
  MaxRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = false;
  MinRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void AllKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = false;
  AllRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void AnyKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out) {
  bool reduce_all = false;
  AnyRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(
    mean, CPU, ALL_LAYOUT, phi::MeanKernel, float, double, bool) {}

PD_REGISTER_KERNEL(sum,
                   CPU,
                   ALL_LAYOUT,
                   phi::SumKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(
    prod, CPU, ALL_LAYOUT, phi::ProdKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    max, CPU, ALL_LAYOUT, phi::MaxKernel, float, double, int, int64_t) {}
PD_REGISTER_KERNEL(
    min, CPU, ALL_LAYOUT, phi::MinKernel, float, double, int, int64_t) {}
PD_REGISTER_KERNEL(all, CPU, ALL_LAYOUT, phi::AllKernel, bool) {}
PD_REGISTER_KERNEL(any, CPU, ALL_LAYOUT, phi::AnyKernel, bool) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

PD_REGISTER_KERNEL(mean,
                   GPU,
                   ALL_LAYOUT,
                   phi::MeanKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(sum,
                   GPU,
                   ALL_LAYOUT,
                   phi::SumKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(
    prod, GPU, ALL_LAYOUT, phi::ProdKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    max, GPU, ALL_LAYOUT, phi::MaxKernel, float, double, int, int64_t) {}
PD_REGISTER_KERNEL(
    min, GPU, ALL_LAYOUT, phi::MinKernel, float, double, int, int64_t) {}
PD_REGISTER_KERNEL(all, GPU, ALL_LAYOUT, phi::AllKernel, bool) {}
PD_REGISTER_KERNEL(any, GPU, ALL_LAYOUT, phi::AnyKernel, bool) {}
#endif
