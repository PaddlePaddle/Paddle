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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/roll_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void RollGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const ScalarArray& shifts,
                    const std::vector<int64_t>& axis,
                    DenseTensor* x_grad) {
  std::vector<T> out_vec;
  paddle::framework::TensorToVector(out_grad, dev_ctx, &out_vec);

  auto shifts_data = shifts.GetData();
  size_t nums = shifts_data.size();
  DDim input_dim = out_grad.dims();
  auto dims = axis;

  // axis = none, reshape to 1-D tensor
  if (dims.size() == 0) {
    dims.push_back(0l);
    input_dim = phi::Dim<1>(out_vec.size());
  }

  for (size_t i = 0; i < nums; i++) {
    ShiftAlongDim(out_vec.data(), input_dim, dims[i], 0 - shifts_data[i]);
  }

  dev_ctx.template Alloc<T>(x_grad);
  paddle::framework::TensorFromVector(out_vec, dev_ctx, x_grad);
  x_grad->Resize(out_grad.dims());
}

}  // namespace phi

PD_REGISTER_KERNEL(roll_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::RollGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
