//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/math_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

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
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  int axis = -1;
  AddRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  int axis = -1;
  SubtractRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out) {
  int axis = -1;
  DivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  int axis = -1;
  MultiplyRawKernel<T>(dev_ctx, x, y, axis, out);
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

PD_REGISTER_KERNEL(add,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(subtract,
                   CPU,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(divide,
                   CPU,
                   ALL_LAYOUT,
                   phi::DivideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
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
                   phi::dtype::bfloat16) {}

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
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
PD_REGISTER_KERNEL(add,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(subtract,
                   GPU,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(divide,
                   GPU,
                   ALL_LAYOUT,
                   phi::DivideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(multiply,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   complex64,
                   complex128) {}
#endif
