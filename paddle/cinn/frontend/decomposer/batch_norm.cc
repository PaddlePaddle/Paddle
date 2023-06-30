// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/decomposer_registry.h"
#include "paddle/cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace decomposer {

struct BatchNormHelper {
  BatchNormHelper(NetBuilder* net_builder,
                  const std::vector<int>& arg_x_shape,
                  const std::vector<int>& arg_param_shape,
                  std::string data_layout,
                  std::string bn_op_type) {
    CHECK_EQ(arg_x_shape.size(), 4UL)
        << "Only 4-D input tensor is supported, but get " << arg_x_shape.size()
        << "-D input tensor.";

    builder = net_builder;
    x_shape = arg_x_shape;
    param_shape = arg_param_shape;

    if (data_layout == "NCHW") {
      channel_dim = 1;
      reduce_dim = {0, 2, 3};
      element_count = x_shape[0] * x_shape[2] * x_shape[3];
    } else if (data_layout == "NHWC") {
      channel_dim = 3;
      reduce_dim = {0, 1, 2};
      element_count = x_shape[0] * x_shape[1] * x_shape[2];
    } else {
      LOG(FATAL) << data_layout << " setting is not support!";
    }

    num_instructions = builder->size();
    op_type = bn_op_type;
  }

  ~BatchNormHelper() {
    VLOG(4) << op_type << " is decomposed to "
            << builder->size() - num_instructions << " instructions.";
  }

  std::vector<Variable> MeanAndVariance(Variable x) {
    auto mean = Mean(x);
    // variance = reduce_sum(x * x) / nhw - mean * mean, shape = [c], simplified
    // by equation: E(x^2) - [E(x)]^2
    auto variance = Variance(x, mean);
    return {mean, variance};
  }

  std::vector<Variable> GradBiasAndScale(Variable x,
                                         Variable x_mean,
                                         Variable y_grad) {
    auto mean_4d = builder->BroadcastTo(x_mean, x->shape, {channel_dim});
    auto x_mean_diff = builder->Subtract(x, mean_4d);
    // bias_grad = reduce_sum(y_grad), shape = [c]
    auto bias_grad = Reduce(y_grad);
    auto sum_of_y_grad_mul_x_mean_diff =
        Reduce(builder->Multiply(y_grad, x_mean_diff));
    return {bias_grad, sum_of_y_grad_mul_x_mean_diff};
  }

  // mean = reduce_sum(x) / nhw
  Variable Mean(Variable x) {
    auto sum = Reduce(x);
    auto element_count_1d =
        builder->FillConstant(sum->shape,
                              element_count,
                              common::UniqName("element_count"),
                              common::Type2Str(sum->type));
    auto mean = builder->Divide(sum, element_count_1d);
    return mean;
  }

  // variance = reduce_sum(x * x) / nhw - mean * mean
  Variable Variance(Variable x, Variable mean) {
    auto x_square = builder->Multiply(x, builder->Identity(x));
    auto x_square_sum = Reduce(x_square);
    auto element_count_1d =
        builder->FillConstant(x_square_sum->shape,
                              element_count,
                              common::UniqName("element_count"),
                              common::Type2Str(x_square_sum->type));
    auto x_square_mean = builder->Divide(x_square_sum, element_count_1d);
    auto variance = builder->Subtract(
        x_square_mean, builder->Multiply(mean, builder->Identity(mean)));
    return variance;
  }

  // std_variance_inv = rsqrt(variance + epsilon)
  Variable StdVarianceInv1d(Variable variance, float epsilon) {
    auto epsilon_1d = builder->FillConstant(variance->shape,
                                            epsilon,
                                            common::UniqName("epsilon"),
                                            common::Type2Str(variance->type));
    auto std_variance_inv = builder->Rsqrt(builder->Add(variance, epsilon_1d));
    return std_variance_inv;
  }

  // std_variance_inv = rsqrt(variance + epsilon)
  Variable StdVarianceInv4d(Variable variance, float epsilon) {
    auto variance_4d = builder->BroadcastTo(variance, x_shape, {channel_dim});
    auto epsilon_4d =
        builder->FillConstant(variance_4d->shape,
                              epsilon,
                              common::UniqName("epsilon"),
                              common::Type2Str(variance_4d->type));
    auto std_variance_inv_4d =
        builder->Rsqrt(builder->Add(variance_4d, epsilon_4d));
    return std_variance_inv_4d;
  }

  // moving_value = moving_value * momentum + (1.0 - momentum) * saved_value
  // value maybe mean and variance.
  Variable UpdateMeanVariance(Variable moving_value,
                              Variable saved_value,
                              float momentum) {
    auto factor_0 = builder->FillConstant(moving_value->shape,
                                          momentum,
                                          common::UniqName("factor_0"),
                                          common::Type2Str(moving_value->type));
    auto factor_1 = builder->FillConstant(saved_value->shape,
                                          1.0f - momentum,
                                          common::UniqName("factor_1"),
                                          common::Type2Str(saved_value->type));
    auto new_moving_value =
        builder->Add(builder->Multiply(moving_value, factor_0),
                     builder->Multiply(saved_value, factor_1));
    return new_moving_value;
  }

  Variable Reduce(Variable x) { return builder->ReduceSum(x, reduce_dim); }

  NetBuilder* builder{nullptr};
  std::vector<int> x_shape;
  std::vector<int> param_shape;
  std::vector<int> reduce_dim;
  float element_count{0};
  int channel_dim{0};
  std::string op_type;
  int num_instructions{0};
};

void batch_norm_train(const Instruction& instr,
                      const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL)
      << "The number of the given inputs is not equal to the required for op "
      << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 5UL)
      << "The number of the given outputs is not equal to the required for op "
      << instr->op_type;

  auto& x = instr->inputs[0];
  auto& scale = instr->inputs[1];
  auto& bias = instr->inputs[2];
  auto& moving_mean = instr->inputs[3];
  auto& moving_variance = instr->inputs[4];
  CHECK_EQ(scale->type, bias->type);
  CHECK_EQ(scale->type, moving_mean->type);
  CHECK_EQ(scale->type, moving_variance->type);

  float epsilon = instr.GetAttrs<float>("epsilon");
  float momentum = instr.GetAttrs<float>("momentum");
  std::string layout = instr.GetAttrs<std::string>("data_layout");

  NetBuilder* builder = context.builder();
  BatchNormHelper helper(
      builder, x->shape, scale->shape, layout, "batch_norm_train");

  auto mean_variance = helper.MeanAndVariance(x);
  auto mean = mean_variance[0];
  auto variance = mean_variance[1];

  auto mean_4d = builder->BroadcastTo(mean, x->shape, {helper.channel_dim});
  // std_variance_inv = rsqrt(variance + epsilon), shape = [c]
  auto std_variance_inv_4d = helper.StdVarianceInv4d(variance, epsilon);

  // y = scale * (x - mean) * std_variance_inv + bias, shape = [n, c, h, w]
  auto scale_4d = builder->BroadcastTo(scale, x->shape, {helper.channel_dim});
  auto bias_4d = builder->BroadcastTo(bias, x->shape, {helper.channel_dim});
  auto normalized =
      builder->Multiply(builder->Subtract(x, mean_4d), std_variance_inv_4d);
  auto scaled_normalized = builder->Multiply(normalized, scale_4d);
  auto y = builder->Add(scaled_normalized, bias_4d);

  // moving_mean = moving_mean * momentum + (1.0 - momentum) * mean, shape = [c]
  auto new_moving_mean = helper.UpdateMeanVariance(moving_mean, mean, momentum);

  // moving_variance = moving_variance * momentum + (1.0 - momentum) * variance,
  // shape = [c]
  auto new_moving_variance =
      helper.UpdateMeanVariance(moving_variance, variance, momentum);

  context.MapOutToOrigin(y, instr->outputs[0]);
  context.MapOutToOrigin(mean, instr->outputs[1]);
  context.MapOutToOrigin(variance, instr->outputs[2]);
  context.MapOutToOrigin(new_moving_mean, instr->outputs[3]);
  context.MapOutToOrigin(new_moving_variance, instr->outputs[4]);
}

void batch_norm_grad(const Instruction& instr,
                     const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL)
      << " The number of the given inputs is not equal to the required "
      << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 3UL)
      << " The number of the given outputs is not equal to the required"
      << instr->op_type;

  auto& y_grad = instr->inputs[0];
  auto& x = instr->inputs[1];
  auto& scale = instr->inputs[2];
  auto& save_mean = instr->inputs[3];
  auto& save_variance = instr->inputs[4];
  CHECK_EQ(y_grad->type, x->type);
  CHECK_EQ(scale->type, save_mean->type);
  CHECK_EQ(scale->type, save_variance->type);

  auto epsilon = instr.GetAttrs<float>("epsilon");
  auto layout = instr.GetAttrs<std::string>("data_layout");

  NetBuilder* builder = context.builder();
  BatchNormHelper helper(
      builder, x->shape, scale->shape, layout, "batch_norm_grad");

  auto vars = helper.GradBiasAndScale(x, save_mean, y_grad);
  auto bias_grad = vars[0];
  auto sum_of_y_grad_mul_x_mean_diff = vars[1];

  // scale_grad = reduce_sum(y_grad * (x - mean)) * rsqrt(variance + epsilon),
  // shape = [c]
  auto scale_grad =
      builder->Multiply(sum_of_y_grad_mul_x_mean_diff,
                        helper.StdVarianceInv1d(save_variance, epsilon));

  // x_grad = 1/nhw * scale * rsqrt(variance + epsilon) *
  //   (nhw * y_grad - reduce_sum(y_grad) - (x - mean) * reduce_sum(y_grad * (x
  //   - mean)) / (variance + epsilon))
  // => x_grad = tmp0 * (tmp1 - tmp2 - tmp3)
  auto scaled_std_variance_inv =
      builder->Multiply(scale, helper.StdVarianceInv1d(save_variance, epsilon));
  auto element_count_1d =
      builder->FillConstant(scaled_std_variance_inv->shape,
                            helper.element_count,
                            common::UniqName("element_count_1d"),
                            common::Type2Str(scaled_std_variance_inv->type));
  auto tmp0 = builder->BroadcastTo(
      builder->Divide(scaled_std_variance_inv, element_count_1d),
      x->shape,
      {helper.channel_dim});

  auto element_count_4d =
      builder->FillConstant(y_grad->shape,
                            helper.element_count,
                            common::UniqName("element_count_4d"),
                            common::Type2Str(y_grad->type));
  auto tmp1 = builder->Multiply(y_grad, element_count_4d);

  auto tmp2 = builder->BroadcastTo(bias_grad, x->shape, {helper.channel_dim});

  auto mean_4d =
      builder->BroadcastTo(save_mean, x->shape, {helper.channel_dim});
  auto x_mean_diff = builder->Subtract(x, mean_4d);

  auto sum_of_y_grad_mul_x_mean_diff_4d = builder->BroadcastTo(
      sum_of_y_grad_mul_x_mean_diff, x->shape, {helper.channel_dim});
  auto tmp3_0 =
      builder->Multiply(x_mean_diff, sum_of_y_grad_mul_x_mean_diff_4d);
  auto epsilon_1d =
      builder->FillConstant(save_variance->shape,
                            epsilon,
                            common::UniqName("epsilon"),
                            common::Type2Str(save_variance->type));
  auto variance_add_eps = builder->Add(save_variance, epsilon_1d);
  auto variance_add_eps_4d =
      builder->BroadcastTo(variance_add_eps, x->shape, {helper.channel_dim});
  auto tmp3 = builder->Divide(tmp3_0, variance_add_eps_4d);

  auto x_grad = builder->Multiply(
      tmp0, builder->Subtract(builder->Subtract(tmp1, tmp2), tmp3));

  context.MapOutToOrigin(x_grad, instr->outputs[0]);
  context.MapOutToOrigin(scale_grad, instr->outputs[1]);
  context.MapOutToOrigin(bias_grad, instr->outputs[2]);
}

void batch_norm(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 5UL)
      << "The number of the given inputs is not equal to the required for op "
      << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL)
      << "The number of the given outputs is not equal to the required for op "
      << instr->op_type;

  auto& x = instr->inputs[0];
  auto& scale = instr->inputs[1];
  auto& bias = instr->inputs[2];
  auto& moving_mean = instr->inputs[3];
  auto& moving_variance = instr->inputs[4];
  CHECK_EQ(scale->type, bias->type);
  CHECK_EQ(scale->type, moving_mean->type);
  CHECK_EQ(scale->type, moving_variance->type);

  float epsilon = instr.GetAttrs<float>("epsilon");
  float momentum = instr.GetAttrs<float>("momentum");
  std::string layout = instr.GetAttrs<std::string>("data_layout");

  NetBuilder* builder = context.builder();
  BatchNormHelper helper(builder, x->shape, scale->shape, layout, "batch_norm");

  auto mean_4d =
      builder->BroadcastTo(moving_mean, x->shape, {helper.channel_dim});
  // std_variance_inv = rsqrt(variance + epsilon), shape = [c]
  auto std_variance_inv_4d = helper.StdVarianceInv4d(moving_variance, epsilon);

  // y = scale * (x - mean) * std_variance_inv + bias, shape = [n, c, h, w]
  auto scale_4d = builder->BroadcastTo(scale, x->shape, {helper.channel_dim});
  auto bias_4d = builder->BroadcastTo(bias, x->shape, {helper.channel_dim});
  auto normalized =
      builder->Multiply(builder->Subtract(x, mean_4d), std_variance_inv_4d);
  auto scaled_normalized = builder->Multiply(normalized, scale_4d);
  auto y = builder->Add(scaled_normalized, bias_4d);

  context.MapOutToOrigin(y, instr->outputs[0]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(batch_norm_decomposer) {
  CINN_DECOMPOSER_REGISTER(batch_norm, cinn::frontend::decomposer::batch_norm);

  return true;
}

CINN_REGISTER_HELPER(batch_norm_train_decomposer) {
  CINN_DECOMPOSER_REGISTER(batch_norm_train,
                           cinn::frontend::decomposer::batch_norm_train);

  return true;
}

CINN_REGISTER_HELPER(batch_norm_grad_decomposer) {
  CINN_DECOMPOSER_REGISTER(batch_norm_grad,
                           cinn::frontend::decomposer::batch_norm_grad);

  return true;
}
