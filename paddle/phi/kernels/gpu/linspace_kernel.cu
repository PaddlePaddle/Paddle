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

#include "paddle/phi/kernels/linspace_kernel.h"

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
__global__ void LinspaceKernelInner(
    T start, T stop, double step, int64_t size, T* out) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  for (; index < size; index += blockDim.x * gridDim.x) {
    if (index < size / 2) {
      out[index] = static_cast<T>(start + step * index);
    } else {
      out[index] = static_cast<T>(stop - step * (size - index - 1));
    }
  }
}

template <typename T>
__global__ void LinspaceSpecialKernel(T start, T* out) {
  out[0] = static_cast<T>(start);
}

template <typename T, typename Context>
void LinspaceKernel(const Context& ctx,
                    const DenseTensor& start,
                    const DenseTensor& stop,
                    const DenseTensor& number,
                    DataType dtype,
                    DenseTensor* out) {
  auto start_t = phi::funcs::TransDataType(ctx, start, dtype);
  auto stop_t = phi::funcs::TransDataType(ctx, stop, dtype);

  DenseTensor n_start;
  DenseTensor n_stop;
  DenseTensor n_num;
  phi::Copy(ctx, start_t, phi::CPUPlace(), false, &n_start);
  T start_data = n_start.data<T>()[0];
  phi::Copy(ctx, stop_t, phi::CPUPlace(), false, &n_stop);
  T stop_data = n_stop.data<T>()[0];
  phi::Copy(ctx, number, phi::CPUPlace(), false, &n_num);
  int64_t num = static_cast<int64_t>(n_num.data<int32_t>()[0]);

  PADDLE_ENFORCE_GT(
      num,
      0,
      phi::errors::InvalidArgument("The num of linspace op should be larger "
                                   "than 0, but received num is %d",
                                   num));

  out->Resize(phi::make_ddim({num}));
  T* out_data = ctx.template Alloc<T>(out);

  double step = 0;
  auto stream = ctx.stream();
  int block = 512;
  int grid = (num + block - 1) / block;
  if (num != 1) {
    step = (static_cast<double>(stop_data - start_data)) / (num - 1);
    LinspaceKernelInner<T><<<grid, block, 0, stream>>>(
        start_data, stop_data, step, num, out_data);
  } else {
    LinspaceSpecialKernel<T><<<grid, block, 0, stream>>>(start_data, out_data);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(linspace,
                   GPU,
                   ALL_LAYOUT,
                   phi::LinspaceKernel,
                   float,
                   int32_t,
                   int64_t,
                   double) {}
