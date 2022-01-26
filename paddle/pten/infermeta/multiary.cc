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

DenseTensorMeta ConcatInferMeta(const std::vector<DenseTensorMeta>& x_meta,
                                const Scalar& axis_scalar,
                                bool is_runtime) {
  PADDLE_ENFORCE_GE(x_meta.size(),
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The size of input meta vector should be greater"
                        "than 0."));

  int axis = axis_scalar.to<int>();
  // 1. calculate axis
  int rank = x_meta[0].dims.size();
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
  for (auto meta : x_meta) {
    x_dims.push_back(meta.dims);
  }
  pten::DDim out_dim =
      pten::funcs::ComputeAndCheckShape(is_runtime, x_dims, axis);

  return {x_meta[0].dtype, out_dim, x_meta[0].layout};
}

}  // namespace pten
