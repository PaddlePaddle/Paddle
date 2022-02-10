/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/infermeta/multiary.h"

#include "paddle/pten/common/scalar.h"
#include "paddle/pten/kernels/funcs/concat_funcs.h"
namespace pten {

void ConcatInferMeta(const std::vector<MetaTensor>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config) {
  PADDLE_ENFORCE_GE(x.size(),
                    0UL,
                    paddle::platform::errors::InvalidArgument(
                        "The size of input meta vector should be greater"
                        "than 0."));

  int axis = axis_scalar.to<int>();
  // 1. calculate axis
  int rank = x.at(0).dims().size();
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      paddle::platform::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }

  // 2. calculate out dims
  std::vector<pten::DDim> x_dims;
  for (auto& x_t : x) {
    x_dims.push_back(x_t.dims());
  }
  pten::DDim out_dim =
      pten::funcs::ComputeAndCheckShape(config.is_runtime, x_dims, axis);

  out->set_dims(out_dim);
  out->set_dtype(x.at(0).dtype());
  out->set_layout(x.at(0).layout());
}

}  // namespace pten
