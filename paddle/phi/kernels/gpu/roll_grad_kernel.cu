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

#include "paddle/phi/kernels/roll_grad_kernel.h"

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/roll_kernel_impl.h"

namespace phi {

using paddle::platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename Context>
void RollGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const IntArray& shifts,
                    const std::vector<int64_t>& axis,
                    DenseTensor* x_grad) {
  auto* in_data = out_grad.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(x_grad);
  int64_t numel = out_grad.numel();
  auto stream = dev_ctx.stream();

  auto shifts_data = shifts.GetData();
  size_t nums = shifts_data.size();
  auto input_dim = out_grad.dims();
  auto stride_dim = phi::stride(input_dim);

  std::vector<int64_t> strides(nums), sizes(nums);
  if (axis.size() == 0) {
    strides[0] = 1;
    sizes[0] = numel;
    shifts_data[0] = ((-shifts_data[0]) % numel + numel) % numel;
  } else {
    for (size_t i = 0; i < nums; i++) {
      int dim = axis[i] >= 0 ? axis[i] : axis[i] + input_dim.size();
      int64_t size = input_dim[dim];
      if (size != 0) {
        shifts_data[i] = ((-shifts_data[i]) % size + size) % size;
        strides[i] = stride_dim[dim];
        sizes[i] = size;
      }
    }
  }

  switch (nums) {
    CALL_ROLL_CUDA_KERNEL(1);
    CALL_ROLL_CUDA_KERNEL(2);
    CALL_ROLL_CUDA_KERNEL(3);
    CALL_ROLL_CUDA_KERNEL(4);
    CALL_ROLL_CUDA_KERNEL(5);
    CALL_ROLL_CUDA_KERNEL(6);
    CALL_ROLL_CUDA_KERNEL(7);
    CALL_ROLL_CUDA_KERNEL(8);
    CALL_ROLL_CUDA_KERNEL(9);
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "shifts.size() should be less than 10, But received shifts.size() "
          "= %d",
          shifts_data.size()));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(roll_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RollGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
