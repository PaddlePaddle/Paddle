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

#include "paddle/phi/kernels/index_elementwise_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"

namespace phi {
template <typename T, typename Context>
void IndexElementwiseKernel(const Context& ctx,
                            const DenseTensor& x,
                            const std::vector<const DenseTensor*>& index,
                            const std::vector<int64_t>& index_dims,
                            const std::vector<int64_t>& index_stride,
                            DenseTensor* out) {
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64,
      true,
      common::errors::InvalidArgument(
          "Index holds the wrong type, it holds [%s], but "
          "desires to be [%s] or [%s].",
          index_type,
          phi::DataType::INT32,
          phi::DataType::INT64));

  size_t total_nonzero = index[0]->numel();
  auto out_dims = out->dims();
  if (out_dims.size() > 0) {
    out_dims[0] = total_nonzero;
    out->Resize(out_dims);
  }

  if (out->numel() == 0) return;
  ctx.template Alloc<T>(out);

  if (index_type == phi::DataType::INT32) {
    phi::funcs::IndexElementwiseKernel<T, int>(
        ctx, x, index, index_dims, index_stride, out);
  } else if (index_type == phi::DataType::INT64) {
    phi::funcs::IndexElementwiseKernel<T, int64_t>(
        ctx, x, index, index_dims, index_stride, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexElementwiseKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
