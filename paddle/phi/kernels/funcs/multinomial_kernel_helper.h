// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void MultinomialInputChecker(const Context& dev_ctx,
                             const DenseTensor& x,
                             const Scalar& num_samples) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  auto in_dims = x.dims();
  int64_t dim_size = in_dims.size();
  const int64_t num_categories = in_dims[dim_size - 1];
  const int64_t num_distributions = dim_size > 1 ? in_dims[dim_size - 2] : 1;
  auto int_num_samples = num_samples.to<int>();

  phi::DenseTensor cpu_tensor;
  phi::Copy<Context>(dev_ctx, x, phi::CPUPlace(), false, &cpu_tensor);
  T* cpu_in_data = cpu_tensor.data<T>();
  for (int64_t i = 0; i < num_distributions; ++i) {
    int zero_num = 0;
    for (int64_t j = 0; j < num_categories; ++j) {
      T weight = cpu_in_data[i * num_categories + j];
      PADDLE_ENFORCE_GE(
          static_cast<MT>(weight),
          0,
          errors::InvalidArgument(
              "Each element of multinomial'input must >= 0, but got %f.",
              static_cast<MT>(weight)));
      if (weight == static_cast<T>(0)) {
        zero_num++;
      }
    }
    int valid_samples = num_categories - zero_num;
    PADDLE_ENFORCE_LE(
        int_num_samples,
        valid_samples,
        errors::InvalidArgument("When replacement=False, 'num_samples' "
                                "must less than or equal to the number of "
                                "positive item of input"));
  }
}

}  // namespace phi
