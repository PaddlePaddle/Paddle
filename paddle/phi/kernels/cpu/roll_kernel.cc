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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/roll_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void RollKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const ScalarArray& shifts,
                const std::vector<int64_t>& axis,
                DenseTensor* out) {
  std::vector<T> out_vec;
  paddle::framework::TensorToVector(x, dev_ctx, &out_vec);

  auto shifts_data = shifts.GetData();
  size_t nums = shifts_data.size();
  DDim input_dim = x.dims();
  auto dims = axis;

  // axis = none, reshape to 1-D tensor
  if (dims.size() == 0) {
    dims.push_back(0l);
    input_dim = phi::Dim<1>(out_vec.size());
  }

  for (size_t i = 0; i < nums; i++) {
    PADDLE_ENFORCE_EQ(
        dims[i] < input_dim.size() && dims[i] >= (0 - input_dim.size()),
        true,
        phi::errors::OutOfRange(
            "Attr(axis[%d]) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(axis[%d]) = %d.",
            i,
            input_dim.size(),
            input_dim.size() - 1,
            i,
            dims[i]));
    ShiftAlongDim(out_vec.data(), input_dim, dims[i], shifts_data[i]);
  }
  dev_ctx.template Alloc<T>(out);
  paddle::framework::TensorFromVector(out_vec, dev_ctx, out);
  out->Resize(x.dims());
}

}  // namespace phi

PD_REGISTER_KERNEL(roll,
                   CPU,
                   ALL_LAYOUT,
                   phi::RollKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
