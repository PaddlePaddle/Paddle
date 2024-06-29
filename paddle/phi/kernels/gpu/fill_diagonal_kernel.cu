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

#include "paddle/phi/kernels/fill_diagonal_kernel.h"

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
void FillDiagonalKernel(const Context& ctx,
                        const DenseTensor& x,
                        float value,
                        int offset,
                        bool wrap,
                        DenseTensor* out) {
  const int64_t kMaxBlockDim = 512;
  phi::Copy(ctx, x, ctx.GetPlace(), false, out);

  T* out_data = ctx.template Alloc<T>(out);
  auto fill_val = static_cast<T>(value);
  T temp_var = static_cast<T>(fill_val);

  auto size = out->numel();
  auto out_dims = out->dims();
  auto strides = funcs::CalStride(out_dims);

  // The wrap mode supported only the dims equals to 2; In wrap mode, the
  // value will be filled in cycles
  if (!wrap) {
    size = std::min(size, out_dims[1] * out_dims[1]);
  }

  int64_t kBlockDim = std::min(int64_t(size / strides), kMaxBlockDim);
  fill_constant_kernel<T><<<1, kBlockDim, 0>>>(
      size, out_data, strides, offset, temp_var, out_dims[1]);
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal,
                   GPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   bool) {}
