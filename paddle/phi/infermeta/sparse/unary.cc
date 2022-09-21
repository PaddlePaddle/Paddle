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

#include "paddle/phi/infermeta/sparse/unary.h"

#include "paddle/phi/core/infermeta_utils.h"

namespace phi {
namespace sparse {

void IndicesInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims({-1});
  out->set_dtype(DataType::INT32);
  out->set_layout(DataLayout::NCHW);
}

void ValuesInferMeta(const MetaTensor& x, MetaTensor* out) {
  const auto& x_dims = x.dims();
  out->set_dims({-1, x_dims[x_dims.size() - 1]});
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void TransposeInferMeta(const MetaTensor& x,
                        const std::vector<int>& axis,
                        MetaTensor* out) {
  auto x_dims = x.dims();
  size_t x_rank = x_dims.size();
  size_t axis_size = axis.size();

  PADDLE_ENFORCE_EQ(
      x_rank,
      axis_size,
      errors::InvalidArgument("The input tensor's dimension "
                              "should be equal to the axis's size. "
                              "But received input tensor's dimension is %d, "
                              "axis's size is %d",
                              x_rank,
                              axis_size));

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    PADDLE_ENFORCE_GE(
        axis[i],
        0,
        errors::InvalidArgument("The axis should be greater than or equal to 0."
                                "But received %d of axis[%d]",
                                axis[i],
                                i));

    PADDLE_ENFORCE_EQ(
        axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
        true,
        errors::InvalidArgument(
            "Each element of Attribute axis should "
            "be a unique value range from 0 to (dims - 1), "
            "where the dims is the axis's size, "
            "unique value means this axis value can appear only once. "
            "But received axis[%d] is %d, axis_size is %d, "
            "count[axis[%d]] is %d",
            i,
            axis[i],
            axis_size,
            i,
            count[axis[i]]));
  }

  phi::DDim out_dims(x_dims);
  for (size_t i = 0; i < axis_size; ++i) {
    out_dims[i] = x_dims[axis[i]];
  }

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void TransposeGradInferMeta(const MetaTensor& x,
                            const std::vector<int>& axis,
                            MetaTensor* out) {
  std::vector<int> reversed_axis(axis);
  for (size_t i = 0; i < axis.size(); i++) {
    reversed_axis[axis[i]] = i;
  }

  TransposeInferMeta(x, reversed_axis, out);
}

}  // namespace sparse
}  // namespace phi
