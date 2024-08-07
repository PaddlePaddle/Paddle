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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void FillDiagonalKernel(const Context& ctx,
                        const DenseTensor& x,
                        float value,
                        int offset,
                        bool wrap,
                        DenseTensor* out) {
  T temp_var = static_cast<T>(value);

  T* out_data = ctx.template Alloc<T>(out);
  phi::Copy(ctx, x, ctx.GetPlace(), false, out);

  auto out_dims = out->dims();
  auto strides = funcs::CalStride(out_dims);
  auto size = out->numel();

  // The wrap mode supported only the dims equals to 2; In wrap mode, the
  // value will be filled in cycles
  if (!wrap) {
    size = std::min(size, out_dims[1] * out_dims[1]);
  }

  for (int64_t i = 0; i < size; i += strides) {
    // to check if the new position with offset is still in the same line;
    // this modify should not affect across lines.
    // out_dims[1] is also work for tensor with dim>2, for which the dims must
    // be the same number
    if (i % out_dims[1] + offset >= 0 &&
        i % out_dims[1] + offset < out_dims[1]) {
      out_data[i + offset] = temp_var;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal,
                   CPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   bool) {}
