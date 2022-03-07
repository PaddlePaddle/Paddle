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

void AucInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& stat_pos,
                  const MetaTensor& stat_neg,
                  const std::string& curve,
                  int num_thresholds,
                  int slide_steps,
                  MetaTensor* auc,
                  MetaTensor* stat_pos_out,
                  MetaTensor* stat_neg_out,
                  MetaConfig config) {
  auto predict_dims = input.dims();
  auto label_dims = label.dims();
  PADDLE_ENFORCE_GE(
      predict_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The Input(Predict) has not been initialized properly. The "
          "shape of Input(Predict) = [%s], the shape size must be "
          "greater_equal 2.",
          predict_dims));
  auto predict_width = predict_dims[1];
  PADDLE_ENFORCE_NE(
      phi::product(predict_dims),
      0,
      phi::errors::InvalidArgument(
          "The Input(Predict) has not been initialized properly. The "
          "shape of Input(Predict) = [%s], the shape can not involes 0.",
          predict_dims));
  PADDLE_ENFORCE_NE(
      phi::product(label_dims),
      0,
      phi::errors::InvalidArgument(
          "The Input(Label) has not been initialized properly. The "
          "shape of Input(Label) = [%s], the shape can not involes 0.",
          label_dims));
  if (config.is_runtime) {
    PADDLE_ENFORCE_LE(
        predict_width,
        2,
        phi::errors::InvalidArgument("Only support binary classification,"
                                     "prediction dims[1] should be 1 or 2"));
  }
  auto predict_height = input.dims()[0];
  auto label_height = label.dims()[0];

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        predict_height,
        label_height,
        phi::errors::InvalidArgument("Out and Label should have same height."));
  }

  int num_pred_buckets = num_thresholds + 1;

  PADDLE_ENFORCE_GE(
      num_pred_buckets,
      1,
      phi::errors::InvalidArgument("num_thresholds must larger than 1"));
  PADDLE_ENFORCE_GE(
      slide_steps,
      0,
      phi::errors::InvalidArgument("slide_steps must be natural number"));

  auc->set_dims({1});
  auc->set_dtype(DataType::INT64);

  if (slide_steps) {
    stat_pos_out->set_dims({(1 + slide_steps) * num_pred_buckets + 1});
    stat_pos_out->set_dtype(DataType::INT64);
    stat_neg_out->set_dims({(1 + slide_steps) * num_pred_buckets + 1});
    stat_neg_out->set_dtype(DataType::INT64);
  } else {
    stat_pos_out->set_dims({1, num_pred_buckets});
    stat_pos_out->set_dtype(DataType::INT64);
    stat_neg_out->set_dims({1, num_pred_buckets});
    stat_neg_out->set_dtype(DataType::INT64);
  }
}

void AdamaxInferMeta(const MetaTensor& param,
                     const MetaTensor& grad,
                     const MetaTensor& learning_rate,
                     const MetaTensor& moment,
                     const MetaTensor& inf_norm,
                     const MetaTensor& beta1_pow,
                     float beta1,
                     float beta2,
                     float epsilon,
                     MetaTensor* param_out,
                     MetaTensor* moment_out,
                     MetaTensor* inf_norm_out) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_NE(
      product(lr_dims),
      0,
      errors::InvalidArgument("Maybe the Input variable LearningRate has not "
                              "been initialized. You may need to confirm "
                              "if you put exe.run(startup_program) "
                              "after optimizer.minimize function."));
  PADDLE_ENFORCE_EQ(
      product(lr_dims),
      1,
      errors::InvalidArgument("Learning rate should have 1 dimension"));
  auto beta1_pow_dims = beta1_pow.dims();
  PADDLE_ENFORCE_EQ(product(beta1_pow_dims),
                    1,
                    errors::InvalidArgument(
                        "Beta1 power accumulator should have 1 dimension"));
  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      grad.dims(),
      errors::InvalidArgument(
          "Param and Grad input of AdamaxOp should have same dimension"));
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment.dims(),
      errors::InvalidArgument(
          "Param and Moment input of AdamaxOp should have same dimension"));
  PADDLE_ENFORCE_EQ(
      param_dims,
      inf_norm.dims(),
      errors::InvalidArgument(
          "Param and InfNorm input of AdamaxOp should have same dimension"));

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());

  moment_out->set_dims(param_dims);
  moment_out->set_dtype(moment.dtype());

  inf_norm_out->set_dims(param_dims);
  inf_norm_out->set_dtype(inf_norm.dtype());
}

void AdadeltaInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& avg_squared_grad,
                       const MetaTensor& avg_squared_update,
                       float rho,
                       float epsilon,
                       MetaTensor* param_out,
                       MetaTensor* avg_squared_grad_out,
                       MetaTensor* avg_squared_update_out) {
  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      grad.dims(),
      errors::InvalidArgument(
          "Param and grad input of AdadeltaOp should have same dimension."));
  PADDLE_ENFORCE_EQ(
      param_dims,
      avg_squared_grad.dims(),
      errors::InvalidArgument("Param and AvgSquaredGrad input of AdadeltaOp "
                              "should have same dimension"));
  PADDLE_ENFORCE_EQ(
      param_dims,
      avg_squared_update.dims(),
      errors::InvalidArgument("Param and AvgSquaredUpdate input of AdadeltaOp "
                              "should have same dimension"));

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());

  avg_squared_grad_out->set_dims(param_dims);
  avg_squared_grad_out->set_dtype(avg_squared_grad.dtype());

  avg_squared_update_out->set_dims(param_dims);
  avg_squared_update_out->set_dtype(avg_squared_update.dtype());
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
