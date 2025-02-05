// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/legacy/elementwise_add_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_subtract_kernel.h"

namespace phi {

template <typename T, typename Context>
void FusedElementwiseAddKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               int axis,
                               const std::string& fuse_activation UNUSED,
                               float fuse_alpha UNUSED,
                               float fuse_beta UNUSED,
                               float fused_output_scale UNUSED,
                               const std::vector<int>& fused_unsqueeze2_axes
                                   UNUSED,
                               float scale_x UNUSED,
                               float scale_y UNUSED,
                               float scale_out UNUSED,
                               DenseTensor* out) {
  AddRawKernel<T, Context>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FusedElementwiseDivKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               int axis,
                               const std::string& fuse_activation UNUSED,
                               float fuse_alpha UNUSED,
                               float fuse_beta UNUSED,
                               float fused_output_scale UNUSED,
                               const std::vector<int>& fused_unsqueeze2_axes
                                   UNUSED,
                               float scale_x UNUSED,
                               float scale_y UNUSED,
                               float scale_out UNUSED,
                               DenseTensor* out) {
  DivideRawKernel<T, Context>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FusedElementwiseMulKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               int axis,
                               const std::string& fuse_activation UNUSED,
                               float fuse_alpha UNUSED,
                               float fuse_beta UNUSED,
                               float fused_output_scale UNUSED,
                               const std::vector<int>& fused_unsqueeze2_axes
                                   UNUSED,
                               float scale_x UNUSED,
                               float scale_y UNUSED,
                               float scale_out UNUSED,
                               DenseTensor* out) {
  MultiplyRawKernel<T, Context>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FusedElementwiseSubKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               int axis,
                               const std::string& fuse_activation UNUSED,
                               float fuse_alpha UNUSED,
                               float fuse_beta UNUSED,
                               float fused_output_scale UNUSED,
                               const std::vector<int>& fused_unsqueeze2_axes
                                   UNUSED,
                               float scale_x UNUSED,
                               float scale_y UNUSED,
                               float scale_out UNUSED,
                               DenseTensor* out) {
  SubtractRawKernel<T, Context>(dev_ctx, x, y, axis, out);
}
}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(fused_elementwise_add,
                   CPU,
                   ALL_LAYOUT,
                   phi::FusedElementwiseAddKernel,
                   float,
                   double,
                   int,
                   bool,
                   int64_t,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(fused_elementwise_div,
                   CPU,
                   ALL_LAYOUT,
                   phi::FusedElementwiseDivKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(fused_elementwise_mul,
                   CPU,
                   ALL_LAYOUT,
                   phi::FusedElementwiseMulKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(fused_elementwise_sub,
                   CPU,
                   ALL_LAYOUT,
                   phi::FusedElementwiseSubKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}
