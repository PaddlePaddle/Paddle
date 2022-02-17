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

#include "paddle/extension.h"

namespace paddle {

namespace custom_kernel {

// Here we use dot <CPU, ANY, INT8> for test
// This test will fail when this kernel is supported in framework
template <typename T, typename Context>
void Dot(const Context& dev_ctx,
         const paddle::Tensor& x,
         const paddle::Tensor& y,
         paddle::Tensor* out) {
  auto const *x_ptr = x.data<T>(), *x_ptr_ = &x_ptr[0];
  auto const *y_ptr = y.data<T>(), *y_ptr_ = &y_ptr[0];
  auto* z = out->mutable_data<T>(paddle::PlaceType::kCPU);

  // Loop over the total N elements of both operands while sum-reducing every
  // B pairs along the way where B is the dimension of the least ordered axis
  auto shape = x.shape();
  auto const N = x.numel();
  auto const B = shape[shape.size() - 1];

  for (int j = 0; j < N / B; j++) {
    T ss = 0;
    for (int i = 0; i < B; i++) ss += (*x_ptr_++) * (*y_ptr_++);
    z[j] = ss;
  }
}

}  // namespace custom_kernel
}  // namespace paddle

PD_REGISTER_KERNEL(dot, CPU, ALL_LAYOUT, paddle::custom_kernel::Dot, int8_t) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::INT8);
}
