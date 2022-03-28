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

#include "paddle/phi/kernels/allclose_kernel.h"

#include <cmath>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AllCloseKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const Scalar& rtol,
                    const Scalar& atol,
                    bool equal_nan,
                    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      rtol.dtype(),
      DataType::FLOAT64,
      phi::errors::InvalidArgument(
          "Input (Rtol) type must be double, but get %s.", rtol.dtype()));
  PADDLE_ENFORCE_EQ(
      atol.dtype(),
      DataType::FLOAT64,
      phi::errors::InvalidArgument(
          "Input (Atol) type must be double, but get %s.", atol.dtype()));

  auto* in_a = x.data<T>();
  auto* in_b = y.data<T>();
  auto rtol_v = rtol.to<double>();
  auto atol_v = atol.to<double>();
  auto* out_data = dev_ctx.template Alloc<bool>(out);
  *out_data = true;

  auto num = x.numel();
  for (int64_t i = 0; i < num; ++i) {
    const T a = in_a[i], b = in_b[i];
    bool val;
    if (std::isnan(a) || std::isnan(b)) {
      val = equal_nan && std::isnan(a) == std::isnan(b);
    } else {
      T left = (a > b ? a - b : b - a);
      T right = atol_v + (b > 0 ? rtol_v * b : (-rtol_v) * b);
      T diff = (left > right ? left - right : right - left);
      val = a == b || left <= right || diff <= 1e-15;
    }
    *out_data &= val;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    allclose, CPU, ALL_LAYOUT, phi::AllCloseKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
