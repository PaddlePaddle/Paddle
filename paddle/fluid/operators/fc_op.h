/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/fc_functor.h"

namespace paddle {
namespace operators {

inline void FCOutputSize(const framework::DDim& in_dims,
                         const framework::DDim& w_dims,
                         std::vector<int64_t>& out_dims,  // NOLINT
                         int in_num_col_dims,
                         bool padding_weights) {
  auto in_mat_dims = phi::flatten_to_2d(in_dims, in_num_col_dims);
  auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
  PADDLE_ENFORCE_EQ(
      in_mat_dims[1],
      w_dims0,
      platform::errors::InvalidArgument(
          "The input's second dimension and weight's first dimension is "
          "expected to be the same. But received input's second dimension is "
          "%d, input's shape is %s; weight's first dimension is %d, weight's "
          "shape is %s.",
          in_mat_dims[1],
          in_mat_dims,
          w_dims0,
          phi::make_ddim({w_dims0, w_dims1})));

  out_dims.reserve(static_cast<size_t>(in_num_col_dims + 1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    out_dims.push_back(in_dims[i]);
  }
  out_dims.push_back(w_dims1);
}

}  // namespace operators
}  // namespace paddle
