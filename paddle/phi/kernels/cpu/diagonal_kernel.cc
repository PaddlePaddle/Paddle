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

#include "paddle/phi/kernels/diagonal_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diagonal.h"

namespace phi {

template <typename T, typename Context>
void DiagonalKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    DenseTensor* out) {
  auto* input = &x;
  const T* input_data = input->data<T>();
  auto input_dim = vectorize(input->dims());
  auto input_dim_size = input_dim.size();

  auto* output = out;
  T* output_data = dev_ctx.template Alloc<T>(output);
  auto output_dim = vectorize(output->dims());

  const int64_t offset_ = offset;
  int64_t axis1_ = axis1 < 0 ? input_dim_size + axis1 : axis1;
  int64_t axis2_ = axis2 < 0 ? input_dim_size + axis2 : axis2;

  std::vector<int64_t> input_stride = funcs::ComputeDimStride(input_dim);
  std::vector<int64_t> output_stride = funcs::ComputeDimStride(output_dim);

  int64_t numel = input->numel();

  for (int64_t idx = 0; idx < numel; idx++) {
    std::vector<int64_t> idx_dim(input_dim_size);
    int64_t temp = 0;
    for (size_t i = 0; i < input_dim_size; i++) {
      idx_dim[i] = (idx - temp) / input_stride[i];
      temp = temp + idx_dim[i] * input_stride[i];
    }

    int64_t axis1_dim = idx_dim[axis1_];
    int64_t axis2_dim = idx_dim[axis2_];

    idx_dim.erase(idx_dim.begin() + std::max(axis1_, axis2_));
    idx_dim.erase(idx_dim.begin() + std::min(axis1_, axis2_));

    bool flag = false;
    if (offset_ == 0 && axis1_dim == axis2_dim) {
      idx_dim.push_back(axis1_dim);
      flag = true;
    } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
      idx_dim.push_back(axis1_dim);
      flag = true;
    } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
      idx_dim.push_back(axis2_dim);
      flag = true;
    }
    if (flag) {
      int64_t idx_output = 0;
      for (size_t i = 0; i < idx_dim.size(); i++) {
        idx_output = idx_output + idx_dim[i] * output_stride[i];
      }
      output_data[idx_output] = input_data[idx];
    }
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(diagonal,
                   CPU,
                   ALL_LAYOUT,
                   phi::DiagonalKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   bool) {}
