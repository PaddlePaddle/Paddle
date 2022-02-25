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

#include "paddle/phi/infermeta/multiary.h"

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
namespace phi {

void BilinearTensorProductInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& weight,
                                    paddle::optional<const MetaTensor&> bias,
                                    MetaTensor* out,
                                    MetaConfig config) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto weight_dims = weight.dims();

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2UL,
      errors::InvalidArgument("The input(X) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      y_dims.size(),
      2UL,
      errors::InvalidArgument("The input(Y) must be a 2D Tensor."));
  PADDLE_ENFORCE_EQ(
      weight_dims.size(),
      3UL,
      errors::InvalidArgument(
          "Expected the input(Weight) is a 3D tensor. But received %dD tensor.",
          weight_dims.size()));
  if (config.is_runtime || (x_dims[0] > 0 && y_dims[0] > 0)) {
    PADDLE_ENFORCE_EQ(x_dims[0],
                      y_dims[0],
                      errors::InvalidArgument(
                          "The first dimension(batch_size) of input(X) must be "
                          "equal to the first dimension of the input(Y)."));
  }
  PADDLE_ENFORCE_EQ(x_dims[1],
                    weight_dims[1],
                    errors::InvalidArgument(
                        "The second dimension of input(X) must be equal to "
                        "the second dimension of the input(Weight)."));
  PADDLE_ENFORCE_EQ(y_dims[1],
                    weight_dims[2],
                    errors::InvalidArgument(
                        "The second dimension of input(Y) must be equal to "
                        "the third dimension of the input(Weight)."));

  if (bias.get_ptr()) {
    auto bias_dims = bias->dims();
    PADDLE_ENFORCE_EQ(bias_dims.size(),
                      2UL,
                      errors::InvalidArgument(
                          "The Input(Bias) must be a 2-D tensor with "
                          "the 2nd dimension fixed to 1 (a row vector)."));
    PADDLE_ENFORCE_EQ(bias_dims[0],
                      1UL,
                      errors::InvalidArgument(
                          "The Input(Bias) must be a 2-D tensor with "
                          "the 2nd dimension fixed to 1 (a row vector)."));
    PADDLE_ENFORCE_EQ(bias_dims[1],
                      weight_dims[0],
                      errors::InvalidArgument(
                          "The second dimension of input(Bias) must be equal "
                          "to the first dimension of the input(Weight)."));
  }

  out->set_dims({x_dims[0], weight_dims[0]});
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ConcatInferMeta(const std::vector<MetaTensor>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config) {
  PADDLE_ENFORCE_GE(x.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "The size of input meta vector should be greater"
                        "than 0."));

  int axis = axis_scalar.to<int>();
  // 1. calculate axis
  int rank = x.at(0).dims().size();
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }

  // 2. calculate out dims
  std::vector<phi::DDim> x_dims;
  for (auto& x_t : x) {
    x_dims.push_back(x_t.dims());
  }
  phi::DDim out_dim =
      phi::funcs::ComputeAndCheckShape(config.is_runtime, x_dims, axis);

  out->set_dims(out_dim);
  out->set_dtype(x.at(0).dtype());
  out->set_layout(x.at(0).layout());
}

}  // namespace phi
