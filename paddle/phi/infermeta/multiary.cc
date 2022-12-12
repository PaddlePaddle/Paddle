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

#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"
namespace phi {

std::vector<DDim> GetMetaTensorsDim(
    const std::vector<const MetaTensor*>& tensors) {
  std::vector<DDim> dims;
  dims.reserve(tensors.size());
  for (const MetaTensor* tensor : tensors) {
    dims.emplace_back(tensor->dims());
  }
  return dims;
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

void AdagradInferMeta(const MetaTensor& param,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      float epsilon,
                      MetaTensor* param_out,
                      MetaTensor* moment_out) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(
      phi::product(lr_dims),
      1,
      phi::errors::InvalidArgument("LearningRate should have one element"));
  auto param_dims = param.dims();

  PADDLE_ENFORCE_EQ(
      param_dims,
      moment.dims(),
      phi::errors::InvalidArgument("Param and Moment input of AdagradOp "
                                   "should have the same dimension."));

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());
  moment_out->set_dims(param_dims);
  moment_out->set_dtype(moment.dtype());
}

void AdamInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   const Scalar& beta1,
                   const Scalar& beta2,
                   const Scalar& epsilon,
                   bool lazy_mode,
                   int64_t min_row_size_to_use_multithread,
                   bool multi_precision,
                   bool use_global_beta_pow,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(
      phi::product(lr_dims),
      1,
      errors::InvalidArgument(
          "The number of LearningRate shall be 1, but received %d. Maybe "
          "the Input variable LearningRate has not "
          "been initialized. You may need to confirm "
          "if you put exe.run(startup_program) "
          "after optimizer.minimize function.",
          phi::product(lr_dims)));
  auto beta1_pow_dims = beta1_pow.dims();
  VLOG(3) << "dims of Beta1Pow : [" << beta1_pow_dims << "]";
  PADDLE_ENFORCE_GE(phi::product(beta1_pow_dims),
                    1,
                    errors::InvalidArgument(
                        "The size of Beta1 power accumulator should be greater "
                        "than 0, but received %d.",
                        phi::product(beta1_pow_dims)));
  auto beta2_pow_dims = beta2_pow.dims();
  VLOG(3) << "dims of Beta2Pow : [" << beta2_pow_dims << "]";
  PADDLE_ENFORCE_GE(phi::product(beta2_pow_dims),
                    1,
                    errors::InvalidArgument(
                        "The size of Beta2 power accumulator should be greater "
                        "than 0, but received %d.",
                        phi::product(beta2_pow_dims)));

  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment1.dims(),
      errors::InvalidArgument(
          "Param and Moment1 input of AdamOp should have same dimension. But "
          "received Param dims: [%s], Moment1 dims: [%s].",
          param_dims,
          moment1.dims()));
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment2.dims(),
      errors::InvalidArgument(
          "Param and Moment2 input of AdamOp should have same dimension. But "
          "received Param dims: [%s], Moment2 dims: [%s].",
          param_dims,
          moment2.dims()));

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());

  moment1_out->set_dims(param_dims);
  moment1_out->set_dtype(moment1.dtype());
  moment2_out->set_dims(param_dims);
  moment2_out->set_dtype(moment2.dtype());

  beta1_pow_out->set_dims(beta1_pow_dims);
  beta1_pow_out->set_dtype(beta1_pow.dtype());
  beta2_pow_out->set_dims(beta2_pow_dims);
  beta2_pow_out->set_dtype(beta2_pow.dtype());
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

void AdamwInferMeta(const MetaTensor& param,
                    const MetaTensor& grad,
                    const MetaTensor& learning_rate,
                    const MetaTensor& moment1,
                    const MetaTensor& moment2,
                    const MetaTensor& beta1_pow,
                    const MetaTensor& beta2_pow,
                    const MetaTensor& master_param,
                    const MetaTensor& skip_update,
                    const Scalar& beta1,
                    const Scalar& beta2,
                    const Scalar& epsilon,
                    float lr_ratio,
                    float coeff,
                    bool with_decay,
                    bool lazy_mode,
                    int64_t min_row_size_to_use_multithread,
                    bool multi_precision,
                    bool use_global_beta_pow,
                    MetaTensor* param_out,
                    MetaTensor* moment1_out,
                    MetaTensor* moment2_out,
                    MetaTensor* beta1_pow_out,
                    MetaTensor* beta2_pow_out,
                    MetaTensor* master_param_outs) {
  AdamInferMeta(param,
                grad,
                learning_rate,
                moment1,
                moment2,
                beta1_pow,
                beta2_pow,
                master_param,
                skip_update,
                beta1,
                beta2,
                epsilon,
                lazy_mode,
                min_row_size_to_use_multithread,
                multi_precision,
                use_global_beta_pow,
                param_out,
                moment1_out,
                moment2_out,
                beta1_pow_out,
                beta2_pow_out,
                master_param_outs);
}

void AddNInferMeta(const std::vector<const MetaTensor*>& x,
                   MetaTensor* out,
                   MetaConfig config) {
  auto N = x.size();
  PADDLE_ENFORCE_GT(
      N,
      0,
      phi::errors::InvalidArgument(
          "The input tensor X's dimensions of SumOp "
          "should be larger than 0. But received X's dimensions %d.",
          N));
  if (N == 1) {
    VLOG(3) << "Warning: SumOp have only one input, may waste memory";
  }

  phi::DDim in_dim({0});
  for (size_t i = 0; i < x.size(); ++i) {
    auto x_dim = x[i]->dims();
    // x_dim.size() == 1 means the real dim of selected rows is [0]
    if (x[i]->is_selected_rows() && x_dim.size() == 1) {
      continue;
    }
    // for zero-sized tensor
    if (phi::product(x_dim) == 0) {
      continue;
    }
    // for 0D tensor
    if (x_dim.size() == 0) {
      continue;
    }
    if (phi::product(in_dim) == 0) {
      in_dim = x_dim;
    } else {
      if (config.is_runtime) {
        PADDLE_ENFORCE_EQ(in_dim,
                          x_dim,
                          phi::errors::InvalidArgument(
                              "The input tensor X of SumOp must"
                              " have same shape. But received X[0]'s shape = "
                              "[%s], X[%d]'s shape = [%s].",
                              in_dim,
                              i,
                              x_dim));
      } else {
        PADDLE_ENFORCE_EQ(
            in_dim.size(),
            x_dim.size(),
            phi::errors::InvalidArgument(
                "The input tensor X of SumOp must have same "
                "dimensions. But received X[0]'s dimensions = %d, X[0]'s "
                "shape = "
                "[%s], X[%d]'s dimensions = %d, X[%d]'s shape = [%s].",
                in_dim.size(),
                in_dim,
                i,
                x_dim.size(),
                i,
                x_dim));
        // if in_dim or x_dim has -1, not check equal
        for (int j = 0; j < x_dim.size(); ++j) {
          if (x_dim[j] == -1 || in_dim[j] == -1) {
            continue;
          }
          PADDLE_ENFORCE_EQ(
              in_dim[j],
              x_dim[j],
              phi::errors::InvalidArgument(
                  "The input tensor X of SumOp must have same shape "
                  "if not -1."
                  "But received X[0]'s shape = [%s], X[%d]'s shape = [%s].",
                  in_dim,
                  i,
                  x_dim));
        }
      }
    }
  }
  out->set_dims(in_dim);
  out->share_lod(*x[0]);
}

// TODO(YuanRisheng) This InferMeta is used in Fluid
//                   and will be deleted in the future.
void AddNTensorArrayInferMeta(const std::vector<const MetaTensor*>& x,
                              MetaTensor* out,
                              MetaConfig config) {
  int64_t max_length = 0;
  bool has_tensor_array = false;
  for (auto input : x) {
    if (input->is_tensor_array()) {
      has_tensor_array = true;
      // if input is lod_tensor_array, dims() will return its size (one element)
      max_length =
          input->dims()[0] > max_length ? input->dims()[0] : max_length;
    }
  }

  if (has_tensor_array) {
    if (out->is_tensor_array()) {
      out->set_dims(make_ddim({max_length}));
    }
  } else {
    AddNInferMeta(x, out, config);
  }
}

void AucInferMeta(const MetaTensor& input,
                  const MetaTensor& label,
                  const MetaTensor& stat_pos,
                  const MetaTensor& stat_neg,
                  const MetaTensor& ins_tag_weight,
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

void AverageAccumulatesInferMeta(const MetaTensor& param,
                                 const MetaTensor& in_sum_1,
                                 const MetaTensor& in_sum_2,
                                 const MetaTensor& in_sum_3,
                                 const MetaTensor& in_num_accumulates,
                                 const MetaTensor& in_old_num_accumulates,
                                 const MetaTensor& in_num_updates,
                                 float average_window,
                                 int64_t max_average_window,
                                 int64_t min_average_window,
                                 MetaTensor* out_sum_1,
                                 MetaTensor* out_sum_2,
                                 MetaTensor* out_sum_3,
                                 MetaTensor* out_num_accumulates,
                                 MetaTensor* out_old_num_accumulates,
                                 MetaTensor* out_num_updates) {
  // auto in_dim = param.dims;
  PADDLE_ENFORCE_NE(
      out_sum_1,
      nullptr,
      errors::NotFound(
          "Output(out_sum_1) of AverageAccumulates should not be null."));
  PADDLE_ENFORCE_NE(
      out_sum_2,
      nullptr,
      errors::NotFound(
          "Output(out_sum_2) of AverageAccumulates should not be null."));
  PADDLE_ENFORCE_NE(
      out_sum_3,
      nullptr,
      errors::NotFound(
          "Output(out_sum_3) of AverageAccumulates should not be null."));
  PADDLE_ENFORCE_NE(out_num_accumulates,
                    nullptr,
                    errors::NotFound("Output(out_num_accumulates) of "
                                     "AverageAccumulates should not be null."));

  PADDLE_ENFORCE_NE(out_old_num_accumulates,
                    nullptr,
                    errors::NotFound("Output(out_old_num_accumulates) of "
                                     "AverageAccumulates should not be null."));

  PADDLE_ENFORCE_NE(
      out_num_updates,
      nullptr,
      errors::NotFound(
          "Output(out_num_updates) of AverageAccumulates should not be null."));

  out_sum_1->set_dims(in_sum_1.dims());
  out_sum_1->set_dtype(in_sum_1.dtype());
  out_sum_2->set_dims(in_sum_2.dims());
  out_sum_2->set_dtype(in_sum_2.dtype());
  out_sum_3->set_dims(in_sum_3.dims());
  out_sum_3->set_dtype(in_sum_3.dtype());
  out_num_accumulates->set_dims({1});
  out_num_accumulates->set_dtype(in_num_accumulates.dtype());
  out_old_num_accumulates->set_dims({1});
  out_old_num_accumulates->set_dtype(in_old_num_accumulates.dtype());
  out_num_updates->set_dims({1});
  out_num_updates->set_dtype(in_num_updates.dtype());
}

void BatchNormInferMeta(const MetaTensor& x,
                        const MetaTensor& mean,
                        const MetaTensor& variance,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        bool is_test,
                        float momentum,
                        float epsilon,
                        const std::string& data_layout_str,
                        bool use_global_stats,
                        bool trainable_statistics,
                        MetaTensor* y,
                        MetaTensor* mean_out,
                        MetaTensor* variance_out,
                        MetaTensor* saved_mean,
                        MetaTensor* saved_variance,
                        MetaTensor* reserve_space,
                        MetaConfig config) {
  const auto x_dims = x.dims();
  for (int i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_EQ(
        (x_dims[i] == -1) || (x_dims[i] > 0),
        true,
        phi::errors::InvalidArgument(
            "Each dimension of input tensor is expected to be -1 or a "
            "positive number, but received %d. Input's shape is [%s].",
            x_dims[i],
            x_dims));
  }

  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input "
          "X must greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input X "
          "must smaller than or equal to 5. But received: the shape of input X "
          "= [%s], the dimension of input X = [%d]",
          x_dims,
          x_dims.size()));

  const int64_t C = ((config.is_run_mkldnn_kernel == true) ||
                             (data_layout == DataLayout::kNCHW)
                         ? x_dims[1]
                         : x_dims[x_dims.size() - 1]);
  auto scale_dim = scale.dims();
  auto bias_dim = bias.dims();

  PADDLE_ENFORCE_EQ(
      scale_dim.size(),
      1UL,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of scale must equal to 1."
          "But received: the shape of scale is [%s], the dimension "
          "of scale is [%d]",
          scale_dim,
          scale_dim.size()));
  PADDLE_ENFORCE_EQ(bias_dim.size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "ShapeError: the dimension of bias must equal to 1."
                        "But received: the shape of bias is [%s],the dimension "
                        "of bias is [%d]",
                        bias_dim,
                        bias_dim.size()));

  bool check = true;
  if ((!config.is_runtime) &&
      (phi::product(scale_dim) <= 0 || phi::product(bias_dim) <= 0)) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(scale_dim[0],
                      C,
                      phi::errors::InvalidArgument(
                          "ShapeError: the shape of scale must equal to [%d]"
                          "But received: the shape of scale is [%d]",
                          C,
                          scale_dim[0]));
    PADDLE_ENFORCE_EQ(bias_dim[0],
                      C,
                      phi::errors::InvalidArgument(
                          "ShapeError: the shape of bias must equal to [%d]"
                          "But received: the shape of bias is [%d]",
                          C,
                          bias_dim[0]));
  }
  y->set_dims(x_dims);
  mean_out->set_dims({C});
  variance_out->set_dims({C});
  if (saved_mean) {
    saved_mean->set_dims({C});
  }
  if (saved_variance) {
    saved_variance->set_dims({C});
  }
  if (reserve_space) {
    reserve_space->set_dims({-1});
  }
  y->share_lod(x);
  y->set_dtype(x.dtype());
}

void BatchNormInferInferMeta(const MetaTensor& x,
                             const MetaTensor& mean,
                             const MetaTensor& variance,
                             const MetaTensor& scale,
                             const MetaTensor& bias,
                             float momentum,
                             float epsilon,
                             const std::string& data_layout,
                             MetaTensor* y,
                             MetaTensor* mean_out,
                             MetaTensor* variance_out,
                             MetaConfig config) {
  BatchNormInferMeta(x,
                     mean,
                     variance,
                     scale,
                     bias,
                     /*is_test=*/true,
                     momentum,
                     epsilon,
                     data_layout,
                     /*use_global_stats=*/false,
                     /*trainable_statistics=*/false,
                     y,
                     mean_out,
                     variance_out,
                     /*saved_mean=*/nullptr,
                     /*saved_variance=*/nullptr,
                     /*reserve_space=*/nullptr,
                     config);
}

void BilinearTensorProductInferMeta(const MetaTensor& x,
                                    const MetaTensor& y,
                                    const MetaTensor& weight,
                                    const MetaTensor& bias,
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

  if (bias) {
    auto bias_dims = bias.dims();
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

void BroadcastTensorsInferMeta(const std::vector<const MetaTensor*>& x,
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

void CheckFiniteAndUnscaleInferMeta(const std::vector<const MetaTensor*>& xs,
                                    const MetaTensor& scale,
                                    std::vector<MetaTensor*> outs,
                                    MetaTensor* found_infinite) {
  PADDLE_ENFORCE_EQ(
      xs.size(),
      outs.size(),
      phi::errors::InvalidArgument(
          "The input(X) and output(Out) should have same size in "
          "Operator(check_finite_and_unscale), size of input(X) is %d "
          "and size of output(Out) is %d.",
          xs.size(),
          outs.size()));
  for (size_t i = 0; i < xs.size(); ++i) {
    outs[i]->set_dims(xs[i]->dims());
    outs[i]->set_dtype(xs[i]->dtype());
  }
  found_infinite->set_dims({1});
  found_infinite->set_dtype(DataType::BOOL);
}

void CoalesceTensorInferMeta(const std::vector<const MetaTensor*>& input,
                             DataType dtype,
                             bool copy_data,
                             bool set_constant,
                             bool persist_output,
                             float constant,
                             bool use_align,
                             int align_size,
                             int size_of_dtype,
                             const std::vector<int64_t>& concated_shapes,
                             const std::vector<int64_t>& concated_ranks,
                             std::vector<MetaTensor*> output,
                             MetaTensor* fused_output,
                             MetaConfig config) {
  if (config.is_runtime) {
    return;
  }
  if (size_of_dtype == -1) {
    size_of_dtype = paddle::experimental::SizeOf(dtype);
  }

  auto alignment = [](size_t size, size_t align_size) {
    size_t remaining = size % align_size;
    auto aligned_size = remaining == 0 ? size : size + (align_size - remaining);
    VLOG(4) << remaining << " " << size << " " << align_size << " "
            << aligned_size;
    return aligned_size;
  };
  VLOG(4) << "align_size: " << align_size;
  if (use_align && align_size > 0) {
    int64_t numel = 0;

    for (size_t i = 0; i < input.size(); ++i) {
      const auto& dim = input[i]->dims();
      auto size = phi::product(dim);
      auto len = use_align
                     ? alignment(static_cast<size_t>(size) * size_of_dtype,
                                 align_size) /
                           size_of_dtype
                     : static_cast<size_t>(size);
      numel += len;
    }
    if (fused_output) {
      fused_output->set_dims(phi::make_ddim({numel}));
      fused_output->set_dtype(dtype);
      VLOG(4) << "fused_output size:" << phi::make_ddim({numel});
    }
  }
}

void CheckMemoryContinueInferMeta(const std::vector<const MetaTensor*>& input,
                                  MetaTensor* output,
                                  std::vector<MetaTensor*> xout,
                                  MetaConfig config) {
  if (config.is_runtime) {
    return;
  }
  int64_t numel = 0;
  for (size_t i = 0; i < input.size(); ++i) {
    const auto& dim = input[i]->dims();
    auto size = phi::product(dim);
    auto len = size * paddle::experimental::SizeOf(input[i]->dtype());
    numel += len;
  }
  output->set_dims(phi::make_ddim({numel}));
  output->set_dtype(phi::DataType::INT8);
}

void ConcatInferMeta(const std::vector<const MetaTensor*>& x,
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
      !rank || (axis >= -rank && axis < rank),
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  axis = rank ? axis : 0;
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

inline int ConvOutputSize(
    int input_size, int filter_size, int dilation, int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size,
      0,
      phi::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But received: output's size is %d. The output's size is computed by "
          "((input_size + 2 * padding - (dilation * (filter_size - 1) + 1)) / "
          "stride + 1), where input_size is %d, padding is %d, "
          "filter_size is %d, dilation is %d, stride is %d.",
          output_size,
          input_size,
          padding,
          filter_size,
          dilation,
          stride));

  return output_size;
}

void DeformableConvInferMeta(const MetaTensor& x,
                             const MetaTensor& offset,
                             const MetaTensor& filter,
                             const MetaTensor& mask,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings,
                             const std::vector<int>& dilations,
                             int deformable_groups,
                             int groups,
                             int im2col_step,
                             MetaTensor* out,
                             MetaConfig config) {
  auto in_dims = x.dims();
  auto offset_dims = offset.dims();
  auto filter_dims = filter.dims();

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      phi::errors::InvalidArgument("Conv input should be 4-D tensor, get %u",
                                   in_dims.size()));
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    filter_dims.size(),
                    phi::errors::InvalidArgument(
                        "Conv input dimension and filter dimension should be "
                        "the same. The difference is [%d]: [%d]",
                        in_dims.size(),
                        filter_dims.size()));
  PADDLE_ENFORCE_EQ(in_dims.size() - strides.size(),
                    2U,
                    phi::errors::InvalidArgument(
                        "Conv input dimension and strides "
                        "dimension should be consistent. But received input "
                        "dimension:[%d], strides dimension:[%d]",
                        in_dims.size(),
                        strides.size()));
  PADDLE_ENFORCE_EQ(paddings.size(),
                    strides.size(),
                    phi::errors::InvalidArgument(
                        "Conv paddings dimension and Conv strides dimension "
                        "should be the same. The difference is [%d]: [%d]",
                        paddings.size(),
                        strides.size()));

  PADDLE_ENFORCE_EQ(
      in_dims[1],
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input channels should be equal to filter "
          "channels * groups. The difference is [%d]: [%d]",
          in_dims[1],
          filter_dims[1] * groups));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      phi::errors::InvalidArgument(
          "The number of output channels should be divided by groups. But "
          "received output channels:[%d], groups:[%d]",
          filter_dims[0],
          groups));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % deformable_groups,
      0,
      phi::errors::InvalidArgument(
          "The number of output channels should be "
          "divided by deformable groups. The difference is [%d]: [%d]",
          filter_dims[0] % groups,
          0));

  if (in_dims[0] > im2col_step) {
    PADDLE_ENFORCE_EQ(
        in_dims[0] % im2col_step,
        0U,
        phi::errors::InvalidArgument(
            "Input batchsize must be smaller than or divide im2col_step. But "
            "received Input batchsize:[%d], im2col_step:[%d]",
            in_dims[0],
            im2col_step));
  }

  for (size_t i = 0; i < strides.size(); ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0U,
        phi::errors::InvalidArgument("stride %d size incorrect", i));
  }
  for (size_t i = 0; i < dilations.size(); ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0U,
        phi::errors::InvalidArgument("dilation %d size incorrect", i));
  }

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < strides.size(); ++i) {
    if (!config.is_runtime &&
        (in_dims[i + 2] <= 0 || filter_dims[i + 2] <= 0)) {
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(ConvOutputSize(in_dims[i + 2],
                                            filter_dims[i + 2],
                                            dilations[i],
                                            paddings[i],
                                            strides[i]));
    }
  }

  PADDLE_ENFORCE_EQ(
      output_shape[1] % deformable_groups,
      0U,
      phi::errors::InvalidArgument(
          "output num_filter must divide deformable group size. But received "
          "output num_filter:[%d], deformable group size:[%d]",
          output_shape[1],
          deformable_groups));

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(output_shape[2],
                      offset_dims[2],
                      phi::errors::InvalidArgument(
                          "output height must equal to offset map height. "
                          "The difference is [%d]: [%d]",
                          output_shape[2],
                          offset_dims[2]));
    PADDLE_ENFORCE_EQ(output_shape[3],
                      offset_dims[3],
                      phi::errors::InvalidArgument(
                          "output width must equal to offset map width. The "
                          "difference is [%d]: [%d]",
                          output_shape[3],
                          offset_dims[3]));

    PADDLE_ENFORCE_EQ(offset_dims[1] % (filter_dims[2] * filter_dims[3]),
                      0U,
                      phi::errors::InvalidArgument(
                          "offset filter must divide deformable group size. "
                          "But received [%d]: [%d]",
                          offset_dims[1],
                          filter_dims[2] * filter_dims[3]));
    PADDLE_ENFORCE_EQ(
        offset_dims[1] / (2 * filter_dims[2] * filter_dims[3]),
        deformable_groups,
        phi::errors::InvalidArgument(
            "offset filter must divide deformable group size. But received "
            "[%d]: [%d]",
            offset_dims[1] / (2 * filter_dims[2] * filter_dims[3]),
            deformable_groups));

    if (mask) {
      auto mask_dims = mask.dims();
      PADDLE_ENFORCE_EQ(output_shape[2],
                        mask_dims[2],
                        phi::errors::InvalidArgument(
                            "output height must equal to mask map height. The "
                            "difference is [%d] vs [%d]",
                            output_shape[2],
                            mask_dims[2]));
      PADDLE_ENFORCE_EQ(output_shape[3],
                        mask_dims[3],
                        phi::errors::InvalidArgument(
                            "output width must equal to mask map width. The "
                            "difference is [%d] vs [%d]",
                            output_shape[3],
                            mask_dims[3]));

      PADDLE_ENFORCE_EQ(mask_dims[1] % (filter_dims[2] * filter_dims[3]),
                        0U,
                        phi::errors::InvalidArgument(
                            "mask filter must divide deformable group size. "
                            "But received [%d]: [%d]",
                            mask_dims[1],
                            filter_dims[2] * filter_dims[3]));
      PADDLE_ENFORCE_EQ(mask_dims[1] / (filter_dims[2] * filter_dims[3]),
                        deformable_groups,
                        phi::errors::InvalidArgument(
                            "mask filter must divide deformable group size. "
                            "But received [%d]: [%d]",
                            mask_dims[1] / (filter_dims[2] * filter_dims[3]),
                            deformable_groups));
    }
  }

  out->set_dims(phi::make_ddim(output_shape));
  out->set_dtype(x.dtype());
}

void EditDistanceInferMeta(const MetaTensor& hyps,
                           const MetaTensor& refs,
                           const MetaTensor& hypslength,
                           const MetaTensor& refslength,
                           bool normalized,
                           MetaTensor* sequencenum,
                           MetaTensor* out) {
  auto hyp_dims = hyps.dims();
  auto ref_dims = refs.dims();

  if (hypslength && refslength) {
    auto hyp_length_dims = hypslength.dims();
    auto ref_length_dims = refslength.dims();

    PADDLE_ENFORCE_EQ(
        hyp_dims.size() == 2 && ref_dims.size() == 2 &&
            hyp_dims[0] == ref_dims[0],
        true,
        errors::InvalidArgument(
            "Input(hyps) and Input(refs) must be 2-D Tensors with "
            "identical first dimension. But received Input(Hyps): "
            "input rank %u, input shape [%s]; received Input(Refs): "
            "input rank %u, input shape [%s]",
            hyp_dims.size(),
            hyp_dims,
            ref_dims.size(),
            ref_dims));
    PADDLE_ENFORCE_EQ(
        hyp_length_dims[0] == ref_length_dims[0] &&
            hyp_length_dims[0] == hyp_dims[0],
        true,
        errors::InvalidArgument(
            "Input(hypslength), Input(refslength) and Input(hyps) "
            "should have identical first dimension. But received "
            "Input(hypslength): input rank %u, input shape [%s]; "
            "received Input(refslength): input rank %u, input shape "
            "[%s]; received Input(hyps): input rank %u, input shape "
            "[%s].",
            hyp_length_dims.size(),
            hyp_length_dims,
            ref_length_dims.size(),
            ref_length_dims,
            hyp_dims.size(),
            hyp_dims));
  } else {
    PADDLE_ENFORCE_EQ(
        hyp_dims.size() == 2 && hyp_dims[1] == 1,
        true,
        errors::InvalidArgument(
            "Input(Hyps) must be a 2-D LoDTensor with the 2nd dimension "
            "equal to 1. But received: input rank %u, input shape [%s].",
            hyp_dims.size(),
            hyp_dims));
    PADDLE_ENFORCE_EQ(
        ref_dims.size() == 2 && ref_dims[1] == 1,
        true,
        errors::InvalidArgument(
            "Input(Refs) must be a 2-D LoDTensor with the 2nd dimension "
            "equal to 1. But received: input rank %u, input shape [%s].",
            ref_dims.size(),
            ref_dims));
  }

  out->set_dims(refs.dims());
  out->set_dtype(DataType::FLOAT32);
  sequencenum->set_dims(phi::make_ddim({1}));
  sequencenum->set_dtype(DataType::FLOAT32);
}

void GenerateProposalsV2InferMeta(const MetaTensor& scores,
                                  const MetaTensor& bbox_deltas,
                                  const MetaTensor& im_shape,
                                  const MetaTensor& anchors,
                                  const MetaTensor& variances,
                                  int pre_nms_top_n,
                                  int post_nms_top_n,
                                  float nms_thresh,
                                  float min_size,
                                  float eta,
                                  bool pixel_offset,
                                  MetaTensor* rpn_rois,
                                  MetaTensor* rpn_roi_probs,
                                  MetaTensor* rpn_rois_num) {
  rpn_rois->set_dims(phi::make_ddim({-1, 4}));
  rpn_roi_probs->set_dims(phi::make_ddim({-1, 1}));
}

void GraphReindexInferMeta(const MetaTensor& x,
                           const MetaTensor& neighbors,
                           const MetaTensor& count,
                           const MetaTensor& hashtable_value,
                           const MetaTensor& hashtable_index,
                           bool flag_buffer_hashtable,
                           MetaTensor* reindex_src,
                           MetaTensor* reindex_dst,
                           MetaTensor* out_nodes) {
  auto GraphReindexShapeCheck = [](const phi::DDim& dims,
                                   std::string tensor_name) {
    if (dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          dims[1],
          1,
          phi::errors::InvalidArgument("The last dim of %s should be 1 when it "
                                       "is 2D, but we get %d",
                                       tensor_name,
                                       dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          dims.size(),
          1,
          phi::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              dims.size()));
    }
  };

  GraphReindexShapeCheck(x.dims(), "X");
  GraphReindexShapeCheck(neighbors.dims(), "Neighbors");
  GraphReindexShapeCheck(count.dims(), "Count");
  if (flag_buffer_hashtable) {
    GraphReindexShapeCheck(hashtable_value.dims(), "HashTable_Value");
    GraphReindexShapeCheck(hashtable_index.dims(), "HashTable_Index");
  }

  reindex_src->set_dims({-1});
  reindex_src->set_dtype(neighbors.dtype());
  reindex_dst->set_dims({-1});
  reindex_dst->set_dtype(neighbors.dtype());
  out_nodes->set_dims({-1});
  out_nodes->set_dtype(x.dtype());
}

void GraphSampleNeighborsInferMeta(const MetaTensor& row,
                                   const MetaTensor& col_ptr,
                                   const MetaTensor& x,
                                   const MetaTensor& eids,
                                   const MetaTensor& perm_buffer,
                                   int sample_size,
                                   bool return_eids,
                                   bool flag_perm_buffer,
                                   MetaTensor* out,
                                   MetaTensor* out_count,
                                   MetaTensor* out_eids) {
  // GSN: GraphSampleNeighbors
  auto GSNShapeCheck = [](const phi::DDim& dims, std::string tensor_name) {
    if (dims.size() == 2) {
      PADDLE_ENFORCE_EQ(
          dims[1],
          1,
          phi::errors::InvalidArgument("The last dim of %s should be 1 when it "
                                       "is 2D, but we get %d",
                                       tensor_name,
                                       dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          dims.size(),
          1,
          phi::errors::InvalidArgument(
              "The %s should be 1D, when it is not 2D, but we get %d",
              tensor_name,
              dims.size()));
    }
  };

  GSNShapeCheck(row.dims(), "Row");
  GSNShapeCheck(col_ptr.dims(), "Col_Ptr");
  GSNShapeCheck(x.dims(), "X");
  if (return_eids) {
    GSNShapeCheck(eids.dims(), "Eids");
    out_eids->set_dims({-1});
    out_eids->set_dtype(row.dtype());
  }
  if (flag_perm_buffer) {
    GSNShapeCheck(perm_buffer.dims(), "Perm_Buffer");
  }

  out->set_dims({-1});
  out->set_dtype(row.dtype());
  out_count->set_dims({-1});
  out_count->set_dtype(DataType::INT32);
}

void HSigmoidLossInferMeta(const MetaTensor& x,
                           const MetaTensor& label,
                           const MetaTensor& w,
                           const MetaTensor& bias,
                           const MetaTensor& path,
                           const MetaTensor& code,
                           int num_classes,
                           bool remote_prefetch,
                           bool is_sparse,
                           MetaTensor* out,
                           MetaTensor* pre_out,
                           MetaTensor* w_out) {
  const int64_t input_dims = x.dims()[0];
  const int64_t label_dims = label.dims()[0];
  PADDLE_ENFORCE_EQ(input_dims,
                    label_dims,
                    phi::errors::InvalidArgument(
                        "The first dimension of "
                        "input and label is expected to be the same. "
                        "But received input's first dimension is %d; "
                        "label's first dimension is %d.",
                        input_dims,
                        label_dims));

  std::vector<int64_t> output_shape({input_dims, 1});
  out->set_dims(phi::make_ddim(output_shape));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

static void Interpolate1DInferShapeCheck(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();

  PADDLE_ENFORCE_EQ("linear",
                    interp_method,
                    phi::errors::InvalidArgument(
                        "Interpolation method can only be \"linear\" when"
                        "Input(X) dimension is 3, but got method = %s .",
                        interp_method));
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  for (int i = 0; i < dim_x.size(); ++i) {
    PADDLE_ENFORCE_NE(
        dim_x[i],
        0,
        phi::errors::InvalidArgument("The shape of input(x) should be larged "
                                     "than 0, bug received shape[%d] is %d ",
                                     i,
                                     dim_x[i]));
  }
  if (size_tensor && size_tensor->size() > 0) {
    // top prority size
    auto inputs_name = size_tensor.get();
    PADDLE_ENFORCE_EQ(
        inputs_name.size(),
        1,
        phi::errors::InvalidArgument(
            "Input(SizeTensor)'size of Op(interpolate) must be 1. "
            "Attr(out_shape)'s length must be 1 for 3-D input tensor, but got "
            "size = %d .",
            inputs_name.size()));
    phi::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_w};
    } else {
      dim_out = {dim_x[0], out_w, dim_x[2]};
    }
    output->set_dims(dim_out);
    output->set_dtype(x.dtype());

    return;
  }

  int out_w_tmp;
  if (scale_tensor) {
    auto scale_tensor_dim = scale_tensor.dims();
    PADDLE_ENFORCE_EQ(
        scale_tensor_dim.size(),
        1,
        phi::errors::InvalidArgument(
            "Scale's dimension size must be 1, but got dimension = %d .",
            scale_tensor_dim.size()));
    PADDLE_ENFORCE_EQ(scale_tensor_dim[0],
                      1,
                      phi::errors::InvalidArgument(
                          "Scale's shape must be 1, but got shape = %d .",
                          scale_tensor_dim[0]));
    out_w_tmp = -1;
  } else {
    if (scale.size() > 0) {
      float scale_w = -1;
      scale_w = scale[0];
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      if (scale_w > 0.) {
        // round down
        out_w_tmp = (data_layout == DataLayout::kNCHW
                         ? static_cast<int>(dim_x[2] * scale_w)
                         : static_cast<int>(dim_x[1] * scale_w));
        // protect when input shape is -1
        out_w_tmp = out_w_tmp > 0 ? out_w_tmp : -1;
      }
    } else {
      out_w_tmp = out_w;
    }
  }

  if (out_size && config.is_runtime) {
    auto out_size_dim = out_size.dims();
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(),
        1,
        phi::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got dimention = %d .",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(
        out_size_dim[0],
        1,
        phi::errors::InvalidArgument(
            "OutSize's 0-th dimension's value must be 1, but got value = %d .",
            out_size_dim[0]));

    // dims will be seted in kernel
    output->set_dtype(x.dtype());
    output->share_lod(x);
    return;
  }

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_w_tmp};
  } else {
    dim_out = {dim_x[0], out_w_tmp, dim_x[2]};
  }
  output->set_dims(dim_out);
  output->set_dtype(x.dtype());
}

static void Interpolate2DInferShapeCheck(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();

  PADDLE_ENFORCE(
      "bilinear" == interp_method || "nearest" == interp_method ||
          "bicubic" == interp_method,
      phi::errors::InvalidArgument(
          "Interpolation method can only be \"bilinear\" or \"nearest\" when "
          "Input(X) dimension is 4, but got method = %s.",
          interp_method));
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  for (int i = 0; i < dim_x.size(); ++i) {
    PADDLE_ENFORCE_NE(
        dim_x[i],
        0,
        phi::errors::InvalidArgument("The shape of input(x) should be larged "
                                     "than 0, bug received shape[%d] is %d ",
                                     i,
                                     dim_x[i]));
  }

  if (size_tensor && size_tensor->size()) {
    // top prority size
    auto inputs_name = size_tensor.get();
    PADDLE_ENFORCE_EQ(
        inputs_name.size(),
        2,
        phi::errors::InvalidArgument(
            "Input(SizeTensor)'size of Op(interpolate) must be 2. "
            "Attr(out_shape)'s length must be 2 for 4-D input "
            "tensor, but got size = %d .",
            inputs_name.size()));
    phi::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_h, out_w, dim_x[3]};
    }
    output->set_dims(dim_out);
    output->set_dtype(x.dtype());

    return;
  }

  int out_h_tmp, out_w_tmp;
  if (scale_tensor) {
    auto scale_tensor_dim = scale_tensor.dims();
    PADDLE_ENFORCE_EQ(
        scale_tensor_dim.size(),
        1,
        phi::errors::InvalidArgument(
            "Scale's dimension size must be 1, but got dimension = %d .",
            scale_tensor_dim.size()));
    PADDLE_ENFORCE_EQ(scale_tensor_dim[0] == 2 || scale_tensor_dim[0] == 1,
                      true,
                      phi::errors::InvalidArgument(
                          "Scale's shape must be 2 or 1, but got shape = %d .",
                          scale_tensor_dim[0]));
    out_h_tmp = -1;
    out_w_tmp = -1;
  } else {
    if (scale.size() > 0) {
      float scale_h = -1;
      float scale_w = -1;
      scale_h = scale[0];
      scale_w = scale[1];
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
      if (scale_h > 0. && scale_w > 0.) {
        // round down
        out_h_tmp = (data_layout == DataLayout::kNCHW
                         ? static_cast<int>(dim_x[2] * scale_h)
                         : static_cast<int>(dim_x[1] * scale_h));
        out_w_tmp = (data_layout == DataLayout::kNCHW
                         ? static_cast<int>(dim_x[3] * scale_w)
                         : static_cast<int>(dim_x[2] * scale_w));
        // protect when input shape is -1
        out_h_tmp = out_h_tmp > 0 ? out_h_tmp : -1;
        out_w_tmp = out_w_tmp > 0 ? out_w_tmp : -1;
      }
    } else {
      out_h_tmp = out_h;
      out_w_tmp = out_w;
    }
  }

  if (out_size && config.is_runtime) {
    auto out_size_dim = out_size.dims();
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(),
        1,
        phi::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got dimension = %d .",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(
        out_size_dim[0],
        2,
        phi::errors::InvalidArgument(
            "OutSize's dim[0] must be 2, but got dimention = %d .",
            out_size_dim[0]));
    // dims will be seted in kernel
    output->set_dtype(x.dtype());
    output->share_lod(x);
    return;
  }

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_h_tmp, out_w_tmp};
  } else {
    dim_out = {dim_x[0], out_h_tmp, out_w_tmp, dim_x[3]};
  }

  output->set_dims(dim_out);
  output->set_dtype(x.dtype());
}

static void Interpolate3DInferShapeCheck(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();

  PADDLE_ENFORCE("nearest" == interp_method || "trilinear" == interp_method,
                 phi::errors::InvalidArgument(
                     "Interpolation method can only be \"trilinear\" or "
                     "\"nearest\" when Input(X) "
                     "dimension is 5, but got method = %s .",
                     interp_method));
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  for (int i = 0; i < dim_x.size(); ++i) {
    PADDLE_ENFORCE_NE(
        dim_x[i],
        0,
        phi::errors::InvalidArgument("The shape of input(x) should be larged "
                                     "than 0, bug received shape[%d] is %d ",
                                     i,
                                     dim_x[i]));
  }

  if (size_tensor && size_tensor->size() > 0) {
    // top prority size
    auto inputs_name = size_tensor.get();
    PADDLE_ENFORCE_EQ(
        inputs_name.size(),
        3,
        phi::errors::InvalidArgument(
            "Input(SizeTensor)'s size of Op(interpolate) must be 3. "
            "Attr(out_shape)'s length must be 3 for 5-D input "
            "tensor, but got size = %d .",
            inputs_name.size()));
    phi::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_d, out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_d, out_h, out_w, dim_x[4]};
    }
    output->set_dims(dim_out);
    output->set_dtype(x.dtype());
    return;
  }

  int out_d_tmp, out_h_tmp, out_w_tmp;
  if (scale_tensor) {
    auto scale_tensor_dim = scale_tensor.dims();
    PADDLE_ENFORCE_EQ(
        scale_tensor_dim.size(),
        1,
        phi::errors::InvalidArgument(
            "Scale's dimension size must be 1, but got size = %d .",
            scale_tensor_dim.size()));
    PADDLE_ENFORCE_EQ(scale_tensor_dim[0] == 3 || scale_tensor_dim[0] == 1,
                      true,
                      phi::errors::InvalidArgument(
                          "Scale's shape must be 3 or 1, but got shape = %d .",
                          scale_tensor_dim[0]));
    out_d_tmp = -1;
    out_h_tmp = -1;
    out_w_tmp = -1;
  } else {
    if (scale.size() > 0) {
      float scale_d = -1;
      float scale_h = -1;
      float scale_w = -1;
      scale_d = scale[0];
      scale_h = scale[1];
      scale_w = scale[2];
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_w in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_h in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
      PADDLE_ENFORCE_EQ(
          scale_d > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_d in Attr(scale) of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_d));
      if (scale_d > 0. && scale_h > 0. && scale_w > 0.) {
        // round down
        out_d_tmp = (data_layout == DataLayout::kNCHW
                         ? static_cast<int>(dim_x[2] * scale_d)
                         : static_cast<int>(dim_x[1] * scale_d));
        out_h_tmp = (data_layout == DataLayout::kNCHW
                         ? static_cast<int>(dim_x[3] * scale_h)
                         : static_cast<int>(dim_x[2] * scale_h));
        out_w_tmp = (data_layout == DataLayout::kNCHW
                         ? static_cast<int>(dim_x[4] * scale_w)
                         : static_cast<int>(dim_x[3] * scale_w));
        // protect when input shape is -1
        out_d_tmp = out_d_tmp > 0 ? out_d_tmp : -1;
        out_h_tmp = out_h_tmp > 0 ? out_h_tmp : -1;
        out_w_tmp = out_w_tmp > 0 ? out_w_tmp : -1;
      }
    } else {
      out_d_tmp = out_d;
      out_h_tmp = out_h;
      out_w_tmp = out_w;
    }
  }

  if (out_size && config.is_runtime) {
    auto out_size_dim = out_size.dims();
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(),
        1,
        phi::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got size is %d.",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(out_size_dim[0],
                      3,
                      phi::errors::InvalidArgument(
                          "OutSize's dim[0] must be 3, but got size is %d.",
                          out_size_dim[0]));
    // dims will be seted in kernel
    output->set_dtype(x.dtype());
    output->share_lod(x);
    return;
  }

  phi::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_d_tmp, out_h_tmp, out_w_tmp};
  } else {
    dim_out = {dim_x[0], out_d_tmp, out_h_tmp, out_w_tmp, dim_x[4]};
  }
  output->set_dims(dim_out);
  output->set_dtype(x.dtype());
}

void InterpolateInferMeta(
    const MetaTensor& x,
    const MetaTensor& out_size,
    const paddle::optional<std::vector<const MetaTensor*>>& size_tensor,
    const MetaTensor& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    MetaTensor* output,
    MetaConfig config) {
  auto dim_x = x.dims();  // NCHW format
  PADDLE_ENFORCE(
      dim_x.size() == 3 || dim_x.size() == 4 || dim_x.size() == 5,
      phi::errors::Unimplemented(
          "Input(X) dimension must be 3, 4 or 5, but got dimension = %d .",
          dim_x.size()));
  if (dim_x.size() == 3) {
    // shape check for 1D interpolate for input tensor shape NCHW
    Interpolate1DInferShapeCheck(x,
                                 out_size,
                                 size_tensor,
                                 scale_tensor,
                                 data_layout_str,
                                 out_d,
                                 out_h,
                                 out_w,
                                 scale,
                                 interp_method,
                                 align_corners,
                                 align_mode,
                                 output,
                                 config);
  } else if (dim_x.size() == 4) {
    // shape check for 2D interpolate for input tensor shape NCHW
    Interpolate2DInferShapeCheck(x,
                                 out_size,
                                 size_tensor,
                                 scale_tensor,
                                 data_layout_str,
                                 out_d,
                                 out_h,
                                 out_w,
                                 scale,
                                 interp_method,
                                 align_corners,
                                 align_mode,
                                 output,
                                 config);
  } else {  // dim_x.size() == 5
    // shape check for 3D interpolate for input tensor shape NCDHW
    Interpolate3DInferShapeCheck(x,
                                 out_size,
                                 size_tensor,
                                 scale_tensor,
                                 data_layout_str,
                                 out_d,
                                 out_h,
                                 out_w,
                                 scale,
                                 interp_method,
                                 align_corners,
                                 align_mode,
                                 output,
                                 config);
  }
}

void LambInferMeta(const MetaTensor& param,
                   const MetaTensor& grad,
                   const MetaTensor& learning_rate,
                   const MetaTensor& moment1,
                   const MetaTensor& moment2,
                   const MetaTensor& beta1_pow,
                   const MetaTensor& beta2_pow,
                   const MetaTensor& master_param,
                   const MetaTensor& skip_update,
                   float weight_decay,
                   float beta1,
                   float beta2,
                   float epsilon,
                   bool multi_precision,
                   MetaTensor* param_out,
                   MetaTensor* moment1_out,
                   MetaTensor* moment2_out,
                   MetaTensor* beta1_pow_out,
                   MetaTensor* beta2_pow_out,
                   MetaTensor* master_param_outs) {
  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_NE(
      phi::product(lr_dims),
      0,
      phi::errors::InvalidArgument(
          "The number of LearningRate shall not be 0, but received %d. Maybe "
          "the Input variable LearningRate has not "
          "been initialized. You may need to confirm "
          "if you put exe.run(startup_program) "
          "after optimizer.minimize function.",
          phi::product(lr_dims)));
  PADDLE_ENFORCE_EQ(
      phi::product(lr_dims),
      1,
      phi::errors::InvalidArgument(
          "Learning rate should have 1 dimension, but received %d.",
          phi::product(lr_dims)));
  auto beta1_pow_dims = beta1_pow.dims();
  PADDLE_ENFORCE_GE(phi::product(beta1_pow_dims),
                    1,
                    phi::errors::InvalidArgument(
                        "The size of Beta1 power accumulator should be "
                        "greater than 0, but received %d.",
                        phi::product(beta1_pow_dims)));
  auto beta2_pow_dims = beta2_pow.dims();
  PADDLE_ENFORCE_GE(phi::product(beta2_pow_dims),
                    1,
                    phi::errors::InvalidArgument(
                        "The size of Beta2 power accumulator should be "
                        "greater than 0, but received %d.",
                        phi::product(beta2_pow_dims)));

  auto param_dims = param.dims();
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment1.dims(),
      phi::errors::InvalidArgument(
          "Param and Moment1 input of LambOp should have same dimension. But "
          "received Param dims: [%s], Moment1 dims: [%s].",
          param_dims,
          moment1.dims()));
  PADDLE_ENFORCE_EQ(
      param_dims,
      moment2.dims(),
      errors::InvalidArgument(
          "Param and Moment2 input of AdamOp should have same dimension. But "
          "received Param dims: [%s], Moment2 dims: [%s].",
          param_dims,
          moment2.dims()));

  PADDLE_ENFORCE_NOT_NULL(
      param_out, errors::NotFound("The output param_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      moment1_out,
      errors::NotFound("The output moment1_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      moment2_out,
      errors::NotFound("The output moment2_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      beta1_pow_out,
      errors::NotFound("The output beta1_pow_out can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      beta2_pow_out,
      errors::NotFound("The output beta2_pow_out can not be nullptr"));

  param_out->set_dims(param_dims);
  param_out->set_dtype(param.dtype());

  moment1_out->set_dims(param_dims);
  moment1_out->set_dtype(moment1.dtype());
  moment2_out->set_dims(param_dims);
  moment2_out->set_dtype(moment2.dtype());

  beta1_pow_out->set_dims(beta1_pow_dims);
  beta1_pow_out->set_dtype(beta1_pow.dtype());
  beta2_pow_out->set_dims(beta2_pow_dims);
  beta2_pow_out->set_dtype(beta2_pow.dtype());
}

void LogspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       const MetaTensor& base,
                       MetaTensor* out) {
  auto s_dims = start.dims();
  PADDLE_ENFORCE_EQ(
      (s_dims.size() == 1) && (s_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Start) must be [1],"
                                   "but received input shape is [%s].",
                                   s_dims));
  auto e_dims = stop.dims();
  PADDLE_ENFORCE_EQ(
      (e_dims.size() == 1) && (e_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Stop) must be [1],"
                                   "but received input shape is [%s].",
                                   e_dims));
  auto num_dims = number.dims();
  PADDLE_ENFORCE_EQ(
      (num_dims.size() == 1) && (num_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Num) must be [1],"
                                   "but received input shape is [%s].",
                                   num_dims));
  auto b_dims = base.dims();
  PADDLE_ENFORCE_EQ(
      (b_dims.size() == 1) && (b_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Base) must be [1],"
                                   "but received input shape is [%s].",
                                   b_dims));
  out->set_dims(phi::make_ddim({-1}));
  out->set_dtype(start.dtype());
}

void MergedAdamInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& learning_rate,
    const std::vector<const MetaTensor*>& moment1,
    const std::vector<const MetaTensor*>& moment2,
    const std::vector<const MetaTensor*>& beta1_pow,
    const std::vector<const MetaTensor*>& beta2_pow,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> moment1_out,
    std::vector<MetaTensor*> moment2_out,
    std::vector<MetaTensor*> beta1_pow_out,
    std::vector<MetaTensor*> beta2_pow_out,
    std::vector<MetaTensor*> master_param_out) {}

void MergedMomentumInferMeta(
    const std::vector<const MetaTensor*>& param,
    const std::vector<const MetaTensor*>& grad,
    const std::vector<const MetaTensor*>& velocity,
    const std::vector<const MetaTensor*>& learning_rate,
    const paddle::optional<std::vector<const MetaTensor*>>& master_param,
    float mu,
    bool use_nesterov,
    const std::vector<std::string>& regularization_method,
    const std::vector<float>& regularization_coeff,
    bool multi_precision,
    float rescale_grad,
    std::vector<MetaTensor*> param_out,
    std::vector<MetaTensor*> velocity_out,
    std::vector<MetaTensor*> master_param_out) {}

void MeshgridInferMeta(const std::vector<const MetaTensor*>& inputs,
                       std::vector<MetaTensor*> outputs) {
  const size_t inputs_num = inputs.size();

  auto out_shape = std::vector<int>(inputs_num);

  for (size_t i = 0; i < inputs.size(); i++) {
    out_shape[i] = inputs[i]->dims()[0];
  }
  auto out_dims = phi::make_ddim(std::vector<int>(out_shape));
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i]->set_dims(out_dims);
    outputs[i]->set_dtype(inputs[0]->dtype());
  }
}

void MomentumInferMeta(const MetaTensor& param,
                       const MetaTensor& grad,
                       const MetaTensor& velocity,
                       const MetaTensor& learning_rate,
                       const MetaTensor& master_param,
                       float mu,
                       bool use_nesterov,
                       const std::string& regularization_method,
                       float regularization_coeff,
                       bool multi_precision,
                       float rescale_grad,
                       MetaTensor* param_out,
                       MetaTensor* velocity_out,
                       MetaTensor* master_param_out) {
  PADDLE_ENFORCE_NE(
      param_out,
      nullptr,
      errors::NotFound("Output(ParamOut) of Momentum should not be null."));
  PADDLE_ENFORCE_NE(
      velocity_out,
      nullptr,
      errors::NotFound("Output(VelocityOut) of Momentum should not be null."));

  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_NE(
      phi::product(lr_dims),
      0,
      errors::InvalidArgument("Maybe the Input variable LearningRate has not "
                              "been initialized. You may need to confirm "
                              "if you put exe.run(startup_program) "
                              "after optimizer.minimize function."));
  PADDLE_ENFORCE_EQ(
      phi::product(lr_dims),
      1,
      errors::InvalidArgument("Learning_rate should be a scalar. But Received "
                              "LearningRate's dim [%s]",
                              phi::product(lr_dims)));

  auto param_dim = param.dims();
  param_out->set_dims(param_dim);
  velocity_out->set_dims(param_dim);

  if (master_param_out) {
    master_param_out->set_dims(param_dim);
  }
}

void MultiDotInferMeta(const std::vector<const MetaTensor*>& x,
                       MetaTensor* out) {
  auto inputs_dims = GetMetaTensorsDim(x);

  const size_t inputs_num = inputs_dims.size();
  PADDLE_ENFORCE_GT(
      inputs_num,
      static_cast<size_t>(1),
      phi::errors::InvalidArgument(
          "The number of input tensors in multi_dot op should > 1."));

  const size_t n = inputs_dims.size();
  auto first_dim = inputs_dims[0];

  bool is_vector = false;
  phi::DDim out_dim;

  PADDLE_ENFORCE_LT(
      first_dim.size(),
      static_cast<size_t>(3),
      phi::errors::InvalidArgument(
          "multi_dot: the first input tensor must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the first tensor is 1D of size n view it as a row vector (1, n)
  if (first_dim.size() == 1) {
    first_dim = phi::make_ddim({1, static_cast<int>(first_dim[0])});
    is_vector = true;
  }

  auto last_dim = inputs_dims[n - 1];
  PADDLE_ENFORCE_LT(
      last_dim.size(),
      static_cast<size_t>(3),
      phi::errors::InvalidArgument(
          "the last input tensor of multi_dot must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the last tensor is 1D of size n view it as a column vector (n, 1)
  if (last_dim.size() == 1) {
    last_dim = phi::make_ddim({static_cast<int>(last_dim[0]), 1});
    out_dim = is_vector ? phi::make_ddim({1}) : phi::make_ddim({first_dim[0]});
  } else {
    out_dim = is_vector ? phi::make_ddim({last_dim[1]})
                        : phi::make_ddim({first_dim[0], last_dim[1]});
  }

  auto width = first_dim[1];
  for (size_t i = 1; i < n - 1; i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(),
                      static_cast<size_t>(2),
                      phi::errors::InvalidArgument(
                          "the input tensor of multi_dot op must be 2D."));

    const auto& tmp_dim = inputs_dims[i];
    PADDLE_ENFORCE_EQ(
        tmp_dim[0],
        width,
        phi::errors::InvalidArgument(
            "the input matrix does not meet the multiplication requirements."));
    width = tmp_dim[1];
  }

  PADDLE_ENFORCE_EQ(
      last_dim[0],
      width,
      phi::errors::InvalidArgument(
          "the input matrix does not meet the multiplication requirements."));

  out->set_dims(out_dim);
  out->set_dtype(x.at(0)->dtype());
  out->share_lod(*x.at(0));
}

void MultiplexInferMeta(const std::vector<const MetaTensor*>& ins,
                        const MetaTensor& ids,
                        MetaTensor* out) {
  PADDLE_ENFORCE_NE(
      ins.empty(),
      true,
      phi::errors::InvalidArgument("MultiInput(X) shouldn't be empty."));
  auto ids_dim = ids.dims();
  PADDLE_ENFORCE_EQ(ids_dim.size(),
                    2,
                    phi::errors::PreconditionNotMet(
                        "The index tensor must be a vector with 2 dimensions"));
  PADDLE_ENFORCE_EQ(
      ids_dim[1],
      1,
      phi::errors::PreconditionNotMet(
          "The index tensor must be a vector with batchSize x 1."));

  auto ins_dims = GetMetaTensorsDim(ins);
  auto num_ins = ins_dims.size();
  PADDLE_ENFORCE_GT(
      num_ins,
      1,
      phi::errors::InvalidArgument("multiplex operator should have more than "
                                   "one candidate input tensors."));

  auto in_dim = ins_dims[0];
  PADDLE_ENFORCE_GE(
      in_dim.size(),
      2,
      phi::errors::InvalidArgument(
          "The rank of candidate tensors must be not less than 2."));
  for (size_t i = 1; i < num_ins; i++) {
    auto dim = ins_dims[i];
    PADDLE_ENFORCE_EQ(
        in_dim,
        dim,
        phi::errors::PreconditionNotMet(
            "All the candidate tensors must have the same size."));
  }
  out->set_dims(in_dim);
  out->set_dtype(ins[0]->dtype());
}

void PsroiPoolInferMeta(const MetaTensor& x,
                        const MetaTensor& rois,
                        const MetaTensor& rois_num,
                        int pooled_height,
                        int pooled_width,
                        int output_channels,
                        float spatial_scale,
                        MetaTensor* out) {
  auto input_dims = x.dims();
  auto rois_dims = rois.dims();

  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4,
      errors::InvalidArgument("The format of input tensor is NCHW"));
  PADDLE_ENFORCE_EQ(rois_dims.size(),
                    2,
                    errors::InvalidArgument(
                        "ROIs should be a 2-D LoDTensor of shape (num_rois, 4) "
                        "given as [(x1, y1, x2, y2), ...]"));
  PADDLE_ENFORCE_EQ(rois_dims[1],
                    4,
                    errors::InvalidArgument(
                        "ROIs should be a 2-D LoDTensor of shape (num_rois, 4) "
                        "given as [(x1, y1, x2, y2), ...]"));
  if (rois_num) {
    auto rois_num_dims = rois_num.dims();
    PADDLE_ENFORCE_EQ(
        rois_num_dims.size(),
        1,
        errors::InvalidArgument("The second dimension of RoisNum should "
                                "be 1, but received dimension is %d",
                                rois_num_dims.size()));
  }

  PADDLE_ENFORCE_EQ(
      input_dims[1],
      output_channels * pooled_height * pooled_width,
      errors::InvalidArgument(
          "the channel of X(%d) "
          "should be equal to the product of "
          "output_channels(%d), pooled_height(%d) and pooled_width(%d)",
          input_dims[1],
          output_channels,
          pooled_height,
          pooled_width));

  PADDLE_ENFORCE_GT(pooled_height,
                    0,
                    errors::InvalidArgument(
                        "The pooled output height must be greater than 0"));
  PADDLE_ENFORCE_GT(pooled_width,
                    0,
                    errors::InvalidArgument(
                        "The pooled output width must be greater than 0"));
  PADDLE_ENFORCE_GT(output_channels,
                    1,
                    errors::InvalidArgument(
                        "The pooled output channels must greater than 1"));
  PADDLE_ENFORCE_GT(
      spatial_scale,
      0.0f,
      errors::InvalidArgument("The spatial scale must greater than 0."));

  auto out_dims = input_dims;
  out_dims[0] = rois_dims[0];
  out_dims[1] =
      output_channels;  // input_dims[1] / (pooled_height * pooled_width);
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void RmspropInferMeta(const MetaTensor& param,
                      const MetaTensor& mean_square,
                      const MetaTensor& grad,
                      const MetaTensor& moment,
                      const MetaTensor& learning_rate,
                      const MetaTensor& mean_grad,
                      float epsilon,
                      float decay,
                      float momentum,
                      bool centered,
                      MetaTensor* param_out,
                      MetaTensor* moment_out,
                      MetaTensor* mean_square_out,
                      MetaTensor* mean_grad_out) {
  if (centered) {
    PADDLE_ENFORCE_NOT_NULL(
        mean_grad_out,
        phi::errors::InvalidArgument(
            "Output(MeanGradOut) of RmspropOp should not be null."));
  }

  auto param_dim = param.dims();
  PADDLE_ENFORCE_EQ(param_dim,
                    moment.dims(),
                    phi::errors::InvalidArgument(
                        "Param and Momentum input of RmspropOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and Moment [%s]",
                        param_dim,
                        moment.dims()));
  PADDLE_ENFORCE_EQ(param_dim,
                    mean_square.dims(),
                    phi::errors::InvalidArgument(
                        "Param and Momentum input of RmspropOp "
                        "should have the same dimension. But received "
                        "Param's dim [%s] and MeanSquare [%s]",
                        param_dim,
                        mean_square.dims()));

  auto lr_dim = learning_rate.dims();
  PADDLE_ENFORCE_EQ(phi::product(lr_dim),
                    1,
                    phi::errors::InvalidArgument(
                        "Learning Rate of RmspropOp should be a scalar. But "
                        "received LearningRate's dim [%s]",
                        phi::product(lr_dim)));

  param_out->set_dims(param_dim);
  param_out->set_dtype(param.dtype());
  moment_out->set_dims(param_dim);
  moment_out->set_dtype(moment.dtype());
  mean_square_out->set_dims(param_dim);
  mean_square_out->set_dtype(mean_square.dtype());
  if (centered) {
    mean_grad_out->set_dims(param_dim);
    mean_grad_out->set_dtype(mean_grad.dtype());
  }
}

void RnnInferMeta(const MetaTensor& x,
                  const std::vector<const MetaTensor*>& pre_state,
                  const std::vector<const MetaTensor*>& weight_list,
                  const MetaTensor& sequence_length,
                  float dropout_prob,
                  bool is_bidirec,
                  int input_size,
                  int hidden_size,
                  int num_layers,
                  const std::string& mode,
                  int seed,
                  bool is_test,
                  MetaTensor* out,
                  MetaTensor* dropout_state,
                  std::vector<MetaTensor*> state,
                  MetaTensor* reserve) {
  auto in_dims = x.dims();

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      3,
      phi::errors::InvalidArgument("The rank of Input in RNN  must be 3. But "
                                   "received Input's rank is %d.",
                                   in_dims.size()));

  if (sequence_length) {
    auto seq_dims = sequence_length.dims();
    PADDLE_ENFORCE_EQ(
        in_dims[1],
        seq_dims[0],
        phi::errors::InvalidArgument(
            "The size of SequenceLength has to equal the batch_size. But "
            "received batch_size is %d and the size of SequenceLength is %d.",
            in_dims[1],
            seq_dims[0]));
  }

  PADDLE_ENFORCE_EQ(pre_state[0]->dims().size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The rank of PreState in RNN  must be 3. But "
                        "the received rank is %d.",
                        pre_state[0]->dims().size()));
  size_t i = 0;
  for (; i < pre_state.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        in_dims[1],
        pre_state[i]->dims()[1],
        phi::errors::InvalidArgument(
            "The second dimension size (representing for batch size) of "
            "Input and PreState should be equal. But received %d and %d.",
            in_dims[1],
            pre_state[i]->dims()[1]));
    PADDLE_ENFORCE_EQ(
        pre_state[0]->dims(),
        pre_state[i]->dims(),
        phi::errors::InvalidArgument(
            "The dims of all tensors in PreState should be same. But "
            "received PreState[0] is %s and PreState[%d] is %s.",
            pre_state[0]->dims(),
            i,
            pre_state[i]->dims()));
  }
  size_t num_state = mode == "LSTM" ? 2 : 1;
  PADDLE_ENFORCE_EQ(i,
                    num_state,
                    phi::errors::InvalidArgument(
                        "The number of tensors in PreState of %s should be %d, "
                        "but received %d.",
                        mode,
                        2,
                        i));

  auto out_dims = in_dims;
  out_dims[2] = is_bidirec ? hidden_size * 2 : hidden_size;
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());

  int state_num = pre_state.size();
  for (int i = 0; i < state_num; ++i) {
    state[i]->set_dims(pre_state[i]->dims());
    state[i]->set_dtype(x.dtype());
  }
}

void SgdInferMeta(const MetaTensor& param,
                  const MetaTensor& learning_rate,
                  const MetaTensor& grad,
                  const MetaTensor& master_param,
                  bool multi_precision,
                  MetaTensor* param_out,
                  MetaTensor* master_param_out) {
  PADDLE_ENFORCE_NOT_NULL(param_out,
                          phi::errors::InvalidArgument(
                              "Output(ParamOut) of SGDOp should not be null."));

  auto lr_dims = learning_rate.dims();
  PADDLE_ENFORCE_EQ(phi::product(lr_dims),
                    1,
                    phi::errors::InvalidArgument(
                        "Learning rate should have 1 element. But received "
                        "LearningRate dims [%s]",
                        phi::product(lr_dims)));

  param_out->set_dims(param.dims());
  param_out->set_dtype(param.dtype());
}

void SendUERecvInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         const MetaTensor& src_index,
                         const MetaTensor& dst_index,
                         const std::string& message_op,
                         const std::string& reduce_op,
                         const IntArray& out_size,
                         MetaTensor* out,
                         MetaTensor* dst_count) {
  auto src_index_dims = src_index.dims();
  if (src_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(src_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Src_index should be 1 when it "
                          "is 2D, but we get %d",
                          src_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        src_index_dims.size(),
        1,
        phi::errors::InvalidArgument(
            "The Src_index should be 1D, when it is not 2D, but we get %d",
            src_index_dims.size()));
  }

  auto dst_index_dims = dst_index.dims();
  if (dst_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dst_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Dst_index should be 1 when it "
                          "is 2D, but we get %d",
                          dst_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dst_index_dims.size(),
        1,
        phi::errors::InvalidArgument("The Dst_index should be 1D, "
                                     "when it is not 2D, but we get %d",
                                     dst_index_dims.size()));
  }

  PADDLE_ENFORCE_EQ(src_index_dims[0],
                    dst_index_dims[0],
                    phi::errors::InvalidArgument(
                        "Src_index and Dst_index should have the same shape."));

  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      y_dims[0],
      src_index_dims[0],
      phi::errors::InvalidArgument(
          "Expect Input Y to have size %d as Src_index on the first dimension, "
          "but we get %d",
          src_index_dims[0],
          y_dims[0]));

  auto x_dims = x.dims();
  if (reduce_op == "MEAN") {
    dst_count->set_dims({-1});
    dst_count->set_dtype(DataType::INT32);
  }

  // Infer out's shape according to x and e(need broadcasting condition)
  out->set_dtype(x.dtype());
  auto x_dims1 = phi::vectorize<int>(x_dims);
  auto y_dims1 = phi::vectorize<int>(y_dims);
  std::vector<int> x_dims2(x_dims1.begin() + 1, x_dims1.end());
  std::vector<int> y_dims2(y_dims1.begin() + 1, y_dims1.end());

  int max_dim = std::max(x_dims2.size(), y_dims2.size());
  int axis = std::abs(static_cast<int>(x_dims2.size() - y_dims2.size()));
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  // Only need to broadcast dimensions other than the 0th dimension.
  phi::funcs::GetBroadcastDimsArrays(phi::make_ddim(x_dims2),
                                     phi::make_ddim(y_dims2),
                                     x_dims_array.data(),
                                     y_dims_array.data(),
                                     out_dims_array.data(),
                                     max_dim,
                                     axis);
  out_dims_array.insert(out_dims_array.begin(), -1);
  out->set_dims(phi::make_ddim(out_dims_array));
}

void SendUVInferMeta(const MetaTensor& x,
                     const MetaTensor& y,
                     const MetaTensor& src_index,
                     const MetaTensor& dst_index,
                     const std::string& message_op,
                     MetaTensor* out) {
  auto src_index_dims = src_index.dims();
  if (src_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(src_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Src_index should be 1 when it "
                          "is 2D, but we get %d",
                          src_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        src_index_dims.size(),
        1,
        phi::errors::InvalidArgument(
            "The Src_index should be 1D, when it is not 2D, but we get %d",
            src_index_dims.size()));
  }

  auto dst_index_dims = dst_index.dims();
  if (dst_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dst_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Dst_index should be 1 when it "
                          "is 2D, but we get %d",
                          dst_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dst_index_dims.size(),
        1,
        phi::errors::InvalidArgument("The Dst_index should be 1D, "
                                     "when it is not 2D, but we get %d",
                                     dst_index_dims.size()));
  }

  PADDLE_ENFORCE_EQ(src_index_dims[0],
                    dst_index_dims[0],
                    phi::errors::InvalidArgument(
                        "Src_index and Dst_index should have the same shape."));

  // Infer out's shape according to x and y(need broadcasting condition)
  out->set_dtype(x.dtype());
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto x_dims1 = phi::vectorize<int>(x_dims);
  auto y_dims1 = phi::vectorize<int>(y_dims);
  std::vector<int> x_dims2(x_dims1.begin() + 1, x_dims1.end());
  std::vector<int> y_dims2(y_dims1.begin() + 1, y_dims1.end());
  int max_dim = std::max(x_dims2.size(), y_dims2.size());
  int axis = std::abs(static_cast<int>(x_dims2.size() - y_dims2.size()));
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  // Only need to broadcast dimensions other than the 0th dimension.
  phi::funcs::GetBroadcastDimsArrays(phi::make_ddim(x_dims2),
                                     phi::make_ddim(y_dims2),
                                     x_dims_array.data(),
                                     y_dims_array.data(),
                                     out_dims_array.data(),
                                     max_dim,
                                     axis);
  out_dims_array.insert(out_dims_array.begin(), src_index_dims[0]);
  out->set_dims(phi::make_ddim(out_dims_array));
}

void StackInferMeta(const std::vector<const MetaTensor*>& x,
                    int axis,
                    MetaTensor* out,
                    MetaConfig config) {
  PADDLE_ENFORCE_GT(x.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "Number of Inputs(x) must be larger than 0, but"
                        " received value is:%d.",
                        x.size()));
  const auto& input_dims = GetMetaTensorsDim(x);
  // we reuse concat logic to compute out_dim. we set concat_axis==-1 to check
  // every axis in input_tensors.
  auto out_dim =
      phi::funcs::ComputeAndCheckShape(config.is_runtime, input_dims, -1);
  int rank = input_dims[0].size();
  PADDLE_ENFORCE_GE(
      axis,
      -(rank + 1),
      phi::errors::InvalidArgument(
          "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d, "
          "but received axis is:%d.",
          rank,
          axis));
  PADDLE_ENFORCE_LT(
      axis,
      rank + 1,
      phi::errors::InvalidArgument(
          "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d, "
          "but received axis is:%d",
          rank,
          axis));
  if (axis < 0) axis += (rank + 1);
  auto vec = phi::vectorize<int>(out_dim);
  vec.insert(vec.begin() + axis, input_dims.size());
  out->set_dims(phi::make_ddim(vec));
  out->set_dtype(x.at(0)->dtype());
  out->share_lod(*x.at(0));
}

void UnchangedMultiInferMeta(const std::vector<const MetaTensor*>& x,
                             std::vector<MetaTensor*> out) {
  PADDLE_ENFORCE_EQ(
      x.size(),
      out.size(),
      phi::errors::InvalidArgument(
          "Input's size should be equal to the output's size"
          "but received input size: (%d) does not equals output_size: (%d)",
          x.size(),
          out.size()));
  for (size_t i = 0; i < x.size(); ++i) {
    if (out[i]) {
      out[i]->share_meta(*x[i]);
    }
  }
}

void ShareBufferInferMeta(const std::vector<const MetaTensor*>& xs,
                          const std::vector<bool>& share_dims_and_dtype,
                          std::vector<MetaTensor*> outs,
                          std::vector<MetaTensor*> xouts) {
  if (share_dims_and_dtype.empty()) {
    return;
  }
  PADDLE_ENFORCE_EQ(xs.size(),
                    share_dims_and_dtype.size(),
                    phi::errors::PermissionDenied(
                        "The input(X) and attribute share_dims_and_dtype "
                        "should have the same size, but got size of input(X) "
                        "is %d and size of share_dims_and_dtype is %d.",
                        xs.size(),
                        share_dims_and_dtype.size()));

  for (size_t i = 0; i < xs.size(); ++i) {
    if (share_dims_and_dtype[i]) {
      outs[i]->set_dims(xs[i]->dims());
      outs[i]->set_dtype(xs[i]->dtype());
    }
  }
}

void UpdateLossScalingInferMeta(const std::vector<const MetaTensor*>& xs,
                                const MetaTensor& found_infinite,
                                const MetaTensor& prev_loss_scaling,
                                const MetaTensor& in_good_steps,
                                const MetaTensor& in_bad_steps,
                                std::vector<MetaTensor*> outs,
                                MetaTensor* loss_scaling,
                                MetaTensor* out_good_steps,
                                MetaTensor* out_bad_steps) {
  PADDLE_ENFORCE_EQ(xs.size(),
                    outs.size(),
                    phi::errors::InvalidArgument(
                        "The input(X) and output(Out) should have same size in "
                        "Operator(update_loss_scaling), size of input(X) is %d "
                        "and size of output(Out) is %d.",
                        xs.size(),
                        outs.size()));
  for (size_t i = 0; i < xs.size(); ++i) {
    if (xs[i] != nullptr && outs[i] != nullptr) {
      outs[i]->set_dims(xs[i]->dims());
      outs[i]->set_dtype(xs[i]->dtype());
    }
  }
  loss_scaling->set_dims({1});
  out_good_steps->set_dims({1});
  out_good_steps->set_dtype(DataType::INT32);
  out_bad_steps->set_dims({1});
  out_bad_steps->set_dtype(DataType::INT32);
}

void WarpctcInferMeta(const MetaTensor& logits,
                      const MetaTensor& label,
                      const MetaTensor& logits_length,
                      const MetaTensor& labels_length,
                      int blank,
                      bool norm_by_times,
                      MetaTensor* loss,
                      MetaTensor* warpctcgrad) {
  auto logits_dims = logits.dims();
  int sequence_width = 0;

  if (logits_length) {
    sequence_width = logits_dims[2];
  } else {
    sequence_width =
        static_cast<int>(phi::product(logits_dims) / logits_dims[0]);
  }

  PADDLE_ENFORCE_GE(
      blank,
      0,
      errors::InvalidArgument(
          "The value of Attr(blank) should be in interval [0, %d), "
          "but received %d",
          blank));
  PADDLE_ENFORCE_LT(
      blank,
      sequence_width,
      errors::InvalidArgument(
          "The value of Attr(blank) should be in interval [0, %d), "
          "but received %d",
          blank));

  loss->set_dims({-1, 1});
  loss->set_dtype(logits.dtype());
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

void YoloLossInferMeta(const MetaTensor& x,
                       const MetaTensor& gt_box,
                       const MetaTensor& gt_label,
                       const MetaTensor& gt_score,
                       const std::vector<int>& anchors,
                       const std::vector<int>& anchor_mask,
                       int class_num,
                       float ignore_thresh,
                       int downsample_ratio,
                       bool use_label_smooth,
                       float scale_x_y,
                       MetaTensor* loss,
                       MetaTensor* objectness_mask,
                       MetaTensor* gt_match_mask) {
  auto dim_x = x.dims();
  auto dim_gtbox = gt_box.dims();
  auto dim_gtlabel = gt_label.dims();
  int anchor_num = anchors.size() / 2;
  int mask_num = anchor_mask.size();

  PADDLE_ENFORCE_EQ(dim_x.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input(X) should be a 4-D tensor. But received "
                        "X dimension size(%s)",
                        dim_x.size()));
  PADDLE_ENFORCE_EQ(
      dim_x[2],
      dim_x[3],
      phi::errors::InvalidArgument("Input(X) dim[3] and dim[4] should be euqal."
                                   "But received dim[3](%s) != dim[4](%s)",
                                   dim_x[2],
                                   dim_x[3]));
  PADDLE_ENFORCE_EQ(
      dim_x[1],
      mask_num * (5 + class_num),
      phi::errors::InvalidArgument(
          "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
          "+ class_num))."
          "But received dim[1](%s) != (anchor_mask_number * "
          "(5+class_num)(%s).",
          dim_x[1],
          mask_num * (5 + class_num)));
  PADDLE_ENFORCE_EQ(
      dim_gtbox.size(),
      3,
      phi::errors::InvalidArgument("Input(GTBox) should be a 3-D tensor, but "
                                   "received gtbox dimension size(%s)",
                                   dim_gtbox.size()));
  PADDLE_ENFORCE_EQ(
      dim_gtbox[2],
      4,
      phi::errors::InvalidArgument("Input(GTBox) dim[2] should be 4",
                                   "But receive dim[2](%s) != 5. ",
                                   dim_gtbox[2]));
  PADDLE_ENFORCE_EQ(dim_gtlabel.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "Input(GTLabel) should be a 2-D tensor,"
                        "But received Input(GTLabel) dimension size(%s) != 2.",
                        dim_gtlabel.size()));
  PADDLE_ENFORCE_EQ(
      dim_gtlabel[0],
      dim_gtbox[0],
      phi::errors::InvalidArgument(
          "Input(GTBox) dim[0] and Input(GTLabel) dim[0] should be same,"
          "But received Input(GTLabel) dim[0](%s) != "
          "Input(GTBox) dim[0](%s)",
          dim_gtlabel[0],
          dim_gtbox[0]));
  PADDLE_ENFORCE_EQ(
      dim_gtlabel[1],
      dim_gtbox[1],
      phi::errors::InvalidArgument(
          "Input(GTBox) and Input(GTLabel) dim[1] should be same,"
          "But received Input(GTBox) dim[1](%s) != Input(GTLabel) "
          "dim[1](%s)",
          dim_gtbox[1],
          dim_gtlabel[1]));
  PADDLE_ENFORCE_GT(anchors.size(),
                    0,
                    phi::errors::InvalidArgument(
                        "Attr(anchors) length should be greater then 0."
                        "But received anchors length(%s)",
                        anchors.size()));
  PADDLE_ENFORCE_EQ(anchors.size() % 2,
                    0,
                    phi::errors::InvalidArgument(
                        "Attr(anchors) length should be even integer."
                        "But received anchors length(%s)",
                        anchors.size()));
  for (size_t i = 0; i < anchor_mask.size(); i++) {
    PADDLE_ENFORCE_LT(
        anchor_mask[i],
        anchor_num,
        phi::errors::InvalidArgument(
            "Attr(anchor_mask) should not crossover Attr(anchors)."
            "But received anchor_mask[i](%s) > anchor_num(%s)",
            anchor_mask[i],
            anchor_num));
  }
  PADDLE_ENFORCE_GT(class_num,
                    0,
                    phi::errors::InvalidArgument(
                        "Attr(class_num) should be an integer greater then 0."
                        "But received class_num(%s) < 0",
                        class_num));

  if (gt_score) {
    auto dim_gtscore = gt_score.dims();
    PADDLE_ENFORCE_EQ(
        dim_gtscore.size(),
        2,
        phi::errors::InvalidArgument("Input(GTScore) should be a 2-D tensor"
                                     "But received GTScore dimension(%s)",
                                     dim_gtbox.size()));
    PADDLE_ENFORCE_EQ(
        dim_gtscore[0],
        dim_gtbox[0],
        phi::errors::InvalidArgument(
            "Input(GTBox) and Input(GTScore) dim[0] should be same"
            "But received GTBox dim[0](%s) != GTScore dim[0](%s)",
            dim_gtbox[0],
            dim_gtscore[0]));
    PADDLE_ENFORCE_EQ(
        dim_gtscore[1],
        dim_gtbox[1],
        phi::errors::InvalidArgument(
            "Input(GTBox) and Input(GTScore) dim[1] should be same"
            "But received GTBox dim[1](%s) != GTScore dim[1](%s)",
            dim_gtscore[1],
            dim_gtbox[1]));
  }

  std::vector<int64_t> dim_out({dim_x[0]});
  loss->set_dims(phi::make_ddim(dim_out));
  loss->set_dtype(x.dtype());

  std::vector<int64_t> dim_obj_mask({dim_x[0], mask_num, dim_x[2], dim_x[3]});
  objectness_mask->set_dims(phi::make_ddim(dim_obj_mask));
  objectness_mask->set_dtype(x.dtype());

  std::vector<int64_t> dim_gt_match_mask({dim_gtbox[0], dim_gtbox[1]});
  gt_match_mask->set_dims(phi::make_ddim(dim_gt_match_mask));
  gt_match_mask->set_dtype(x.dtype());
}

}  // namespace phi

PD_REGISTER_INFER_META_FN(batch_norm_infer, phi::BatchNormInferInferMeta);
