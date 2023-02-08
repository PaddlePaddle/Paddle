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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
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
                    const Scalar& start,
                    const Scalar& stop,
                    const Scalar& number,
                    DataType dtype,
                    DenseTensor* out) {
  T start_value = start.to<T>();
  T stop_value = stop.to<T>();
  int64_t num = number.to<int64_t>();
  PADDLE_ENFORCE_GT(
      num,
      0,
      phi::errors::InvalidArgument("The num of linspace op should be larger "
                                   "than 0, but received num is %d",
                                   num));

  out->Resize(phi::make_ddim({num}));
  T* out_data = ctx.template Alloc<T>(out);

  auto stream = ctx.stream();
  if (num != 1) {
    int block = 512;
    int grid = (num + block - 1) / block;
    double step = (static_cast<double>(stop_value - start_value)) / (num - 1);
    LinspaceKernelInner<T><<<grid, block, 0, stream>>>(
        start_value, stop_value, step, num, out_data);
  } else {
    LinspaceSpecialKernel<T><<<1, 1, 0, stream>>>(start_value, out_data);
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
                   double) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
