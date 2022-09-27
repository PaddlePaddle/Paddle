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

#pragma once

#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
struct EyeFunctor {
  EyeFunctor(int64_t num_columns, T* output)
      : num_columns_(num_columns), output_(output) {}

  HOSTDEVICE void operator()(size_t idx) const {
    output_[idx * num_columns_ + idx] = static_cast<T>(1);
  }

  int64_t num_columns_;
  T* output_;
};

template <typename T, typename Context>
void EyeKernel(const Context& ctx,
               const Scalar& num_rows,
               const Scalar& num_columns,
               DataType dtype,
               DenseTensor* out) {
  auto columns = num_columns.to<int64_t>();
  auto rows = num_rows.to<int64_t>();
  if (columns == -1) {
    columns = rows;
  }
  T* out_data = ctx.template Alloc<T>(out);
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(ctx, out, static_cast<T>(0));
  int64_t num_eyes = (std::min)(rows, columns);
  paddle::platform::ForRange<Context> for_range(ctx, num_eyes);
  EyeFunctor<T> functor(columns, out_data);
  for_range(functor);
}

}  // namespace phi
