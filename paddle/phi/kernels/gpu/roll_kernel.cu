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

#include "paddle/phi/kernels/roll_kernel.h"

#include "paddle/common/array.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/roll_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void RollKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& shifts,
                const std::vector<int64_t>& axis,
                DenseTensor* out) {
  if (x.numel() == 0) {
    out->Resize(out->dims());
    dev_ctx.template Alloc<T>(out);
    return;
  }
  auto* in_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto shifts_data = shifts.GetData();
  int rank = shifts_data.size();

  int64_t numel = x.numel();
  auto input_dim = x.dims();
  auto stride_dim = common::stride(input_dim);

  std::vector<int64_t> strides(rank), sizes(rank);
  if (axis.size() == 0) {
    strides[0] = 1;
    sizes[0] = numel;
    shifts_data[0] = (shifts_data[0] % numel + numel) % numel;
  } else {
    for (int i = 0; i < rank; i++) {
      int dim = axis[i] >= 0 ? axis[i] : axis[i] + input_dim.size();
      int64_t size = input_dim[dim];

      if (size != 0) {
        shifts_data[i] = (shifts_data[i] % size + size) % size;
        strides[i] = stride_dim[dim];
        sizes[i] = size;
      }
    }
  }

  LaunchRollKernel<T, Context>(
      dev_ctx, in_data, out_data, rank, numel, shifts_data, strides, sizes);
}

}  // namespace phi

PD_REGISTER_KERNEL(roll,
                   GPU,
                   ALL_LAYOUT,
                   phi::RollKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   bool,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
