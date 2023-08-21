// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/inverse_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void InverseKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto x_shape = vectorize<int>(x.dims());
  PADDLE_ENFORCE_GE(x_shape.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "Input(x) should be a single matrix or batch of "
                        "matrices with shape greater than or equal to 2,"
                        "but the size of its shape is %d.",
                        x_shape.size()));
  PADDLE_ENFORCE_EQ(
      x_shape[x_shape.size() - 1],
      x_shape[x_shape.size() - 2],
      phi::errors::InvalidArgument(
          "Input(x) should be a square matrix or batch of square matrices,"
          "but the matrix dimension is %d x %d.",
          x_shape[x_shape.size() - 1],
          x_shape[x_shape.size() - 2]));

  int actual_byte_size =
      x_shape[x_shape.size() - 1] * x_shape[x_shape.size() - 2] * sizeof(T);
  PADDLE_ENFORCE_LE(
      actual_byte_size,
      8192,
      phi::errors::InvalidArgument(
          "Maximum byte size of Input(x) should be less than or equal to 8192,"
          "but the actual byte size is %d.",
          actual_byte_size));

  int matrix_num = 1;
  for (unsigned int i = 0; i < x_shape.size() - 2; i++) {
    matrix_num *= x_shape[i];
  }
  std::vector<int> matrix_num_vec{matrix_num};
  DenseTensor info = Empty<int>(dev_ctx, IntArray(matrix_num_vec));
  dev_ctx.template Alloc<T>(out);
  int r = xpu::inverse<XPUType>(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x.data<T>()),
                                reinterpret_cast<XPUType*>(out->data<T>()),
                                info.data<int>(),
                                matrix_num,
                                x_shape[x_shape.size() - 1]);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "inverse");

  DenseTensor info_cpu;
  Copy(dev_ctx, info, phi::CPUPlace(), true, &info_cpu);
  auto info_cpu_data = info_cpu.data<int>();
  for (int i = 0; i < matrix_num; i++) {
    PADDLE_ENFORCE_EQ(info_cpu_data[i],
                      0,
                      phi::errors::InvalidArgument(
                          "Matrix[%d] is not inversed correctly!", i));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    inverse, XPU, ALL_LAYOUT, phi::InverseKernel, float, double) {}
