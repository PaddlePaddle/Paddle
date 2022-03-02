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
#include <vector>
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
namespace phi {

std::vector<DDim> GetMetaTensorsDim(const std::vector<MetaTensor*>& tensors) {
  std::vector<DDim> dims;
  dims.reserve(tensors.size());
  for (const MetaTensor* tensor : tensors) {
    dims.emplace_back(tensor->dims());
  }
  return dims;
}

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

void BroadcastTensorsInferMeta(const std::vector<MetaTensor*>& x,
                               std::vector<MetaTensor*> out) {
  int target_rank = 0;
  const auto& input_dims = GetMetaTensorsDim(x);

  // 1. Find Output rank = max(Inputs rank)
  for (const auto& input_ddim : input_dims) {
    target_rank = std::max(target_rank, input_ddim.size());
  }

  PADDLE_ENFORCE_GT(target_rank,
                    0,
                    errors::InvalidArgument("BroadcastTensorsOp requires at "
                                            "least one input tensor to have "
                                            "rank greater than zero"));

  std::vector<int64_t> target_dims(target_rank, 0);
  // 2. Output dim(axis=x) = max(Inputs dim(axis=x))
  for (int index = 0; index < target_rank; index++) {
    // Loop axes in reverse order,
    // For each axis, take the maximum as target size
    // Fill size = 1 if shape vector exhausts
    int target_dim_size = 1;
    for (const auto& input_ddim : input_dims) {
      // Reversed order
      int axis = static_cast<int>(input_ddim.size()) - index - 1;
      int dim_size = 1;
      if (axis >= 0) {
        dim_size = input_ddim[axis];
      }

      if (target_dim_size != 1 && dim_size != 1 &&
          target_dim_size != dim_size) {
        PADDLE_THROW(errors::InvalidArgument(
            "BroadcastTensorsOp inputs does not satisfy bcast semantics, "
            "please check axis = %d in reverse order",
            index));
      }

      // We performed bcast semantics check at python level
      // So input tensors should all have legal shape
      target_dim_size = std::max(target_dim_size, dim_size);
    }
    target_dims[target_rank - index - 1] = target_dim_size;
  }

  // 3. Set Output Dim
  for (size_t i = 0; i < out.size(); i++) {
    out[i]->set_dims(phi::make_ddim(target_dims));
    out[i]->share_lod(*(x[i]));
    out[i]->set_dtype(x[i]->dtype());
  }
}

void ConcatInferMeta(const std::vector<MetaTensor*>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config) {
  PADDLE_ENFORCE_GE(x.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "The size of input meta vector should be greater"
                        "than 0."));
  if (axis_scalar.FromTensor()) {
    auto out_dims =
        phi::make_ddim(std::vector<int>(x.at(0)->dims().size(), -1));
    out->set_dims(out_dims);
    out->set_dtype(x.at(0)->dtype());
    out->set_layout(x.at(0)->layout());
    out->share_lod(*x.at(0));
    return;
  }

  int axis = axis_scalar.to<int>();
  // 1. calculate axis
  int rank = x.at(0)->dims().size();
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
  x_dims.reserve(x.size());
  for (const auto* x_t : x) {
    x_dims.emplace_back(x_t->dims());
  }
  phi::DDim out_dim =
      phi::funcs::ComputeAndCheckShape(config.is_runtime, x_dims, axis);

  out->set_dims(out_dim);
  out->set_dtype(x.at(0)->dtype());
  out->set_layout(x.at(0)->layout());
  out->share_lod(*x.at(0));
}

void WhereInferMeta(const MetaTensor& condition,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    MetaTensor* out) {
  auto cond_dims = condition.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      cond_dims,
      x_dims,
      phi::errors::InvalidArgument(
          "The dims of Inputs(Condition) and Inputs(X) should be same. "
          "But received Condition's shape is [%s], X's shape is [%s]",
          cond_dims,
          x_dims));
  PADDLE_ENFORCE_EQ(x_dims,
                    y_dims,
                    phi::errors::InvalidArgument(
                        "The dims of Inputs(X) and Inputs(Y) should be same. "
                        "But received X's shape is [%s], Y's shape is [%s]",
                        x_dims,
                        y_dims));
  out->share_meta(x);
}

}  // namespace phi
