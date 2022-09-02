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

#include "paddle/phi/infermeta/sparse/binary.h"

namespace phi {
namespace sparse {

inline void GetOutShape(const DDim& x_dims,
                        const std::vector<int>& kernel_sizes,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        DDim* out_dims) {
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument("the shape of x should be (N, D, H, W, C)"));
  PADDLE_ENFORCE_EQ(kernel_sizes.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "the shape of kernel should be (D, H, W, C, OC)"));

  // infer out shape
  (*out_dims)[0] = x_dims[0];
  (*out_dims)[4] = kernel_sizes[4];
  for (int i = 1; i < 4; i++) {
    (*out_dims)[i] = (x_dims[i] + 2 * paddings[i - 1] -
                      dilations[i - 1] * (kernel_sizes[i - 1] - 1) - 1) /
                         strides[i - 1] +
                     1;
  }
}

void Conv3DInferMeta(const MetaTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const bool subm,
                     MetaTensor* out) {
  const auto& x_dims = x.dims();
  DDim out_dims = {1, 1, 1, 1, 1};

  GetOutShape(x_dims, kernel_sizes, paddings, dilations, strides, &out_dims);

  out->set_dtype(x.dtype());
  out->set_dims(out_dims);
  out->set_layout(x.layout());
}

}  // namespace sparse
}  // namespace phi
