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

#include "paddle/phi/kernels/fill_diagonal_grad_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T>
__global__ void fill_constant_kernel(const int64_t featuresize,
                                     T* in_data,
                                     int64_t strides,
                                     int offset,
                                     T fillvar,
                                     int dims) {
  for (int64_t idx = blockIdx.x * featuresize + threadIdx.x;
       idx * strides + offset < (blockIdx.x + 1) * featuresize;
       idx += blockDim.x) {
    // to check if the new position with offset is still in the same line;
    // this modify should not affect across lines.
    // out_dims[1] is also work for tensor with dim>2, for which the dims must
    // be the same number
    if ((idx * strides) % dims + offset < dims &&
        (idx * strides) % dims + offset >= 0) {
      in_data[idx * strides + offset] = fillvar;
    }
  }
}

template <typename T, typename Context>
void FillDiagonalGradKernel(const Context& ctx,
                            const DenseTensor& out_grad,
                            float value,
                            int offset,
                            bool wrap,
                            DenseTensor* x_grad) {
  const int64_t kMaxBlockDim = 512;
  auto* in_data = ctx.template Alloc<T>(x_grad);

  phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);

  auto size = x_grad->numel();
  auto out_dims = x_grad->dims();
  auto strides = funcs::CalStride(out_dims);

  auto wrapsize = std::min(size, out_dims[1] * out_dims[1]);
  // The wrap mode supported only the dims equals to 2; In wrap mode, the
  // value will be filled in cycles
  if (wrap) {
    wrapsize = size;
  }

  int64_t kBlockDim = std::min(int64_t(size), kMaxBlockDim);
  fill_constant_kernel<T><<<1, kBlockDim, 0>>>(
      wrapsize, in_data, strides, offset, T(0), out_dims[1]);
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   bool) {}
