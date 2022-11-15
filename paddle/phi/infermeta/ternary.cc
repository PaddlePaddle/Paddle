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

#include "paddle/phi/infermeta/ternary.h"

#include "glog/logging.h"

#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/impl/box_coder.h"

namespace phi {

void AccuracyInferMeta(const MetaTensor& out,
                       const MetaTensor& indice,
                       const MetaTensor& label,
                       MetaTensor* accuracy,
                       MetaTensor* correct,
                       MetaTensor* total,
                       MetaConfig config) {
  auto inference_dim = out.dims();
  auto label_dim = label.dims();
  // Assume indices has same shape as inference, because
  // it's the output of topk.
  PADDLE_ENFORCE_EQ(
      label_dim.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: label's dimensions of AccuracyOp must be 2. "
          "But received label's dimensions = %d, label's shape = [%s]",
          label_dim.size(),
          label_dim));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(label_dim[1],
                      1,
                      phi::errors::InvalidArgument(
                          "ShapeError: label's second dimension of "
                          "AccuracyOp must be 1. But received label's "
                          "second dimension is = %d, label's shape = [%s]",
                          label_dim[1],
                          label_dim));
    PADDLE_ENFORCE_EQ(
        inference_dim[0],
        label_dim[0],
        phi::errors::InvalidArgument(
            "ShapeError: the output's num_rows of AccuracyOp must be"
            " the same as label's num_rows. But received output's "
            "shape = [%s], label's shape = [%s], output's num_rows = %d, "
            "label's "
            "num_rows = %d",
            inference_dim,
            label_dim,
            inference_dim[0],
            label_dim[0]));
  }

  accuracy->set_dims({1});
  accuracy->set_dtype(out.dtype());
  correct->set_dims({1});
  correct->set_dtype(out.dtype());
  total->set_dims({1});
  total->set_dtype(out.dtype());
  accuracy->share_lod(out);
}

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float beta,
                    float alpha,
                    MetaTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  auto ndim_input = input_dims.size();
  auto ndim_x = x_dims.size();
  auto ndim_y = y_dims.size();

  VLOG(3) << "addmm operator input.shape=" << input_dims
          << " x.shape=" << x_dims << " y.shape=" << y_dims << " beta=" << beta
          << " alpha=" << alpha << " ndim_input=" << ndim_input
          << " ndim_x=" << ndim_x << " ndim_y=" << ndim_y;

  PADDLE_ENFORCE_NE(
      product(input_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'input' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(x_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'x' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(y_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'y' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));
  // dim check
  PADDLE_ENFORCE_EQ(ndim_input == 2 || ndim_input == 1,
                    true,
                    errors::InvalidArgument(
                        "The input tensor input's dimension must be 2 or 1. "
                        "But received input's dimension = [%d].",
                        ndim_input));
  PADDLE_ENFORCE_EQ(
      ndim_x,
      2,
      errors::InvalidArgument("The input tensor x's dimension must be 2. "
                              "But received x's dimension = [%d].",
                              ndim_x));
  PADDLE_ENFORCE_EQ(
      ndim_y,
      2,
      errors::InvalidArgument("The input tensor y's dimension must be 2. "
                              "But received y's dimension = [%d].",
                              ndim_y));

  std::vector<int64_t> output_dims;
  output_dims.push_back(x_dims[0]);
  output_dims.push_back(y_dims[1]);

  out->set_dims(make_ddim(output_dims));
  out->share_lod(input);
  out->set_dtype(input.dtype());
}

void BoxCoderInferMeta(const MetaTensor& prior_box,
                       const MetaTensor& prior_box_var,
                       const MetaTensor& target_box,
                       const std::string& code_type,
                       bool box_normalized,
                       int axis,
                       const std::vector<float>& variance,
                       MetaTensor* output_box,
                       MetaConfig config) {
  auto prior_box_dims = prior_box.dims();
  auto target_box_dims = target_box.dims();

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(prior_box_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The rank of Input PriorBox in BoxCoder operator "
                          "must be 2. But received rank = %d",
                          prior_box_dims.size()));
    PADDLE_ENFORCE_EQ(prior_box_dims[1],
                      4,
                      phi::errors::InvalidArgument(
                          "The second dimension of PriorBox in BoxCoder "
                          "operator must be 4. But received dimension = %d",
                          prior_box_dims[1]));
    if (prior_box_var) {
      auto prior_box_var_dims = prior_box_var.dims();
      PADDLE_ENFORCE_EQ(
          prior_box_var_dims.size(),
          2,
          phi::errors::InvalidArgument(
              "The rank of Input(PriorBoxVar) in BoxCoder operator"
              " should be 2. But received rank = %d",
              prior_box_var_dims.size()));
      PADDLE_ENFORCE_EQ(
          prior_box_dims,
          prior_box_var_dims,
          phi::errors::InvalidArgument(
              "The dimension of Input(PriorBoxVar) should be equal to"
              "the dimension of Input(PriorBox) in BoxCoder operator "
              "when the rank is 2."));
    }
  }

  auto box_code_type = phi::funcs::GetBoxCodeType(code_type);
  if (box_code_type == phi::funcs::BoxCodeType::kEncodeCenterSize) {
    PADDLE_ENFORCE_EQ(target_box_dims.size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The rank of Input TargetBox in BoxCoder operator "
                          "must be 2. But received rank is %d",
                          target_box_dims.size()));
    PADDLE_ENFORCE_EQ(target_box_dims[1],
                      4,
                      phi::errors::InvalidArgument(
                          "The second dimension of TargetBox in BoxCoder "
                          "operator is 4. But received dimension is %d",
                          target_box_dims[1]));
    output_box->set_dims({target_box_dims[0], prior_box_dims[0], 4});
  } else if (box_code_type == phi::funcs::BoxCodeType::kDecodeCenterSize) {
    PADDLE_ENFORCE_EQ(target_box_dims.size(),
                      3,
                      phi::errors::InvalidArgument(
                          "The rank of Input TargetBox in BoxCoder "
                          "operator must be 3. But received rank is %d",
                          target_box_dims.size()));
    PADDLE_ENFORCE_EQ(
        axis == 0 || axis == 1,
        true,
        phi::errors::InvalidArgument("axis in BoxCoder operator must be 0 or 1."
                                     "But received axis = %d",
                                     axis));
    if (config.is_runtime) {
      if (axis == 0) {
        PADDLE_ENFORCE_EQ(
            target_box_dims[1],
            prior_box_dims[0],
            phi::errors::InvalidArgument(
                "When axis is 0, The second "
                "dimension of TargetBox in BoxCoder "
                "should be equal to the first dimension of PriorBox."));
      } else if (axis == 1) {
        PADDLE_ENFORCE_EQ(
            target_box_dims[0],
            prior_box_dims[0],
            phi::errors::InvalidArgument(
                "When axis is 1, The first "
                "dimension of TargetBox in BoxCoder "
                "should be equal to the first dimension of PriorBox."));
      }
      PADDLE_ENFORCE_EQ(
          target_box_dims[2],
          prior_box_dims[1],
          phi::errors::InvalidArgument("The third dimension of TargetBox"
                                       " in BoxCoder should be equal to the "
                                       "second dimension of PriorBox."));
    }
    output_box->share_dims(target_box);
  }

  if (box_code_type == phi::funcs::BoxCodeType::kDecodeCenterSize &&
      axis == 1) {
    output_box->share_lod(prior_box);
  } else {
    output_box->share_lod(target_box);
  }
  output_box->set_dtype(target_box.dtype());
}

void ArangeInferMeta(const MetaTensor& start,
                     const MetaTensor& end,
                     const MetaTensor& step,
                     MetaTensor* out) {
  PADDLE_ENFORCE_EQ(phi::product(start.dims()),
                    1,
                    phi::errors::InvalidArgument(
                        "The numel of Input(start) should be 1, but got %d",
                        phi::product(start.dims())));

  PADDLE_ENFORCE_EQ(phi::product(end.dims()),
                    1,
                    phi::errors::InvalidArgument(
                        "The numel of Input(end) should be 1, but got %d",
                        phi::product(end.dims())));

  PADDLE_ENFORCE_EQ(phi::product(step.dims()),
                    1,
                    phi::errors::InvalidArgument(
                        "The numel of Input(step) should be 1, but got %d",
                        phi::product(step.dims())));

  out->set_dims({-1});
  out->set_dtype(start.dtype());
}

void InstanceNormInferMeta(const MetaTensor& x,
                           const MetaTensor& scale,
                           const MetaTensor& bias,
                           float epsilon,
                           MetaTensor* y,
                           MetaTensor* saved_mean,
                           MetaTensor* saved_variance,
                           MetaConfig config) {
  PADDLE_ENFORCE_NE(y,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The y in InstanceNormInferMeta can't be nullptr."));
  const auto x_dims = x.dims();
  PADDLE_ENFORCE_NE(phi::product(x_dims),
                    0,
                    phi::errors::PreconditionNotMet(
                        "The Input variable X has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function."));
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(),
      5,
      phi::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "smaller than or equal to 5, But received: the shape of input "
          "X = [%s], the dimension of input X = [%d]",
          x_dims,
          x_dims.size()));
  auto N = x_dims[0];
  auto C = x_dims[1];
  auto NxC = N * C;
  if (scale) {
    auto scale_dim = scale.dims();
    PADDLE_ENFORCE_EQ(
        scale_dim.size(),
        1UL,
        phi::errors::InvalidArgument(
            "ShapeError: the dimension of scale must equal to 1."
            "But received: the shape of scale is [%s], the dimension "
            "of scale is [%d]",
            scale_dim,
            scale_dim.size()));
    bool check = !((!config.is_runtime) && (phi::product(scale_dim) <= 0));
    if (check) {
      PADDLE_ENFORCE_EQ(scale_dim[0],
                        C,
                        phi::errors::InvalidArgument(
                            "ShapeError: the shape of scale must equal to [%d]"
                            "But received: the shape of scale is [%d]",
                            C,
                            scale_dim[0]));
    }
  }
  if (bias) {
    auto bias_dim = bias.dims();
    PADDLE_ENFORCE_EQ(
        bias_dim.size(),
        1UL,
        phi::errors::InvalidArgument(
            "ShapeError: the dimension of bias must equal to 1."
            "But received: the shape of bias is [%s],the dimension "
            "of bias is [%d]",
            bias_dim,
            bias_dim.size()));
    bool check = !((!config.is_runtime) && (phi::product(bias_dim) <= 0));
    if (check) {
      PADDLE_ENFORCE_EQ(bias_dim[0],
                        C,
                        phi::errors::InvalidArgument(
                            "ShapeError: the shape of bias must equal to [%d]"
                            "But received: the shape of bias is [%d]",
                            C,
                            bias_dim[0]));
    }
  }
  y->set_dims(x_dims);
  y->share_lod(x);
  y->set_dtype(x.dtype());
  y->set_layout(x.layout());
  if (saved_mean) {
    saved_mean->set_dims({NxC});
  }
  if (saved_variance) {
    saved_variance->set_dims({NxC});
  }
}

void GroupNormInferMeta(const MetaTensor& x,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        float epsilon,
                        int groups,
                        const std::string& data_layout_str,
                        MetaTensor* y,
                        MetaTensor* mean,
                        MetaTensor* variance) {
  PADDLE_ENFORCE_NE(y,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The y in GroupNormInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(mean,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "The mean in GroupNormInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(
      variance,
      nullptr,
      phi::errors::InvalidArgument(
          "The variance in GroupNormInferMeta can't be nullptr."));

  auto x_dim = x.dims();
  PADDLE_ENFORCE_GE(
      x_dim.size(),
      2,
      phi::errors::InvalidArgument(
          "The Input(X)'s dimension of Op(group_norm) must be "
          "greater than 1. But received: %u-D Tensor, which shape is [%s].",
          x_dim.size(),
          x_dim));

  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  const int64_t channel_num =
      (data_layout == DataLayout::kNCHW ? x_dim[1] : x_dim[x_dim.size() - 1]);
  auto batch_size = x_dim[0];
  PADDLE_ENFORCE_LE(
      groups,
      channel_num,
      phi::errors::InvalidArgument(
          "The Attr(groups) of Op(group_norm) must be less than or "
          "equal to the number of channels. But received: groups "
          "is [%s], channels is [%s], the Attr(data_layout) "
          "is [%s]. The error may come from wrong data_layout setting.",
          groups,
          channel_num,
          data_layout_str));
  PADDLE_ENFORCE_GE(
      groups,
      1,
      phi::errors::InvalidArgument(
          "The Attr(groups) of Op(group_norm) must be "
          "greater than or equal to 1. But received: groups is [%s].",
          groups));
  PADDLE_ENFORCE_EQ(
      channel_num % groups,
      0,
      phi::errors::InvalidArgument(
          "Expected number of channels in input to be divisible by "
          "num_groups, but got input channel is %d and num_groups is %d",
          channel_num,
          groups));

  if (scale) {
    PADDLE_ENFORCE_EQ(
        scale.dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The Input(Scale) of Op(group_norm) should be 1-D Tensor. "
            "But received: %u-D Tensor, the shape of Input(Scale) is [%s].",
            scale.dims().size(),
            scale.dims()));
    PADDLE_ENFORCE_EQ(
        scale.dims()[0],
        channel_num,
        phi::errors::InvalidArgument(
            "The Input(Scale)'s first dimension size of Op(group_norm) must "
            "be equal to the number of channels. But received: the "
            "Input(Scale)'s first dimension size is [%s], the channels is "
            "[%s], the Attr(data_layout) is [%s]. The error may come "
            "from wrong data_layout setting.",
            scale.dims()[0],
            channel_num,
            data_layout_str));
  }
  if (bias) {
    PADDLE_ENFORCE_EQ(
        bias.dims().size(),
        1UL,
        phi::errors::InvalidArgument(
            "The Input(Bias) of Op(group_norm) should be 1-D Tensor. "
            "But received: %u-D Tensor, the shape of Input(Bias) is [%s].",
            bias.dims().size(),
            bias.dims()));
    PADDLE_ENFORCE_EQ(
        bias.dims()[0],
        channel_num,
        phi::errors::InvalidArgument(
            "The Input(Bias)'s first dimension size of "
            "Op(group_norm) must be equal to the number of channels. "
            "But received: the Input(Bias)'s first dimension size is [%s], "
            "the channels is [%s], the Attr(data_layout) is [%s]. The "
            "error may come from wrong data_layout setting.",
            bias.dims()[0],
            channel_num,
            data_layout_str));
  }
  y->set_dims(x_dim);
  y->set_dtype(x.dtype());
  y->share_lod(x);
  mean->set_dims({batch_size, groups});
  variance->set_dims({batch_size, groups});
}

void LayerNormInferMeta(const MetaTensor& x,
                        const MetaTensor& scale,
                        const MetaTensor& bias,
                        float epsilon,
                        int begin_norm_axis,
                        bool is_test,
                        MetaTensor* out,
                        MetaTensor* mean,
                        MetaTensor* variance,
                        MetaConfig config) {
  auto x_dim = x.dims();
  PADDLE_ENFORCE_LT(
      begin_norm_axis,
      x_dim.size(),
      phi::errors::InvalidArgument(
          "'begin_norm_axis' must be less than the dimensions of X,"
          "But received 'begin_norm_axis' is [%d],"
          "received the dimensions of X is [%d].",
          begin_norm_axis,
          x_dim.size()));

  auto matrix_dim = phi::flatten_to_2d(x_dim, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  if (scale) {
    PADDLE_ENFORCE_EQ(scale.dims().size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The dimensions of Input(Scale) must be 1, but "
                          "received dimensions of"
                          "Input(Scale) is [%d]",
                          scale.dims().size()));
  }

  if (config.is_runtime && scale) {
    PADDLE_ENFORCE_EQ(
        scale.dims()[0],
        right,
        phi::errors::InvalidArgument(
            "The first dimension value of Input(Scale) must equal to be the"
            "second dimension value of the flattened 2D matrix of Input(X),"
            "But received the first dimension value of Input(Scale) is"
            "[%d], the second dimension value of the flattened 2D matrix of"
            " Input(Scale) is [%d].",
            scale.dims()[0],
            right));
  }
  if (bias) {
    PADDLE_ENFORCE_EQ(bias.dims().size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The dimensions of Input(Bias) must be 1, but "
                          "received dimensions of"
                          "Input(Bias) is [%d]",
                          bias.dims().size()));
  }
  if (config.is_runtime && bias) {
    PADDLE_ENFORCE_EQ(
        bias.dims()[0],
        right,
        phi::errors::InvalidArgument(
            "The first dimension value of Input(Bias) must equal to be the"
            "second dimension value of the flattened 2D matrix of Input(X),"
            "But received the first dimension value of Input(Bias) is"
            "[%d], the second dimension value of the flattened 2D matrix of"
            " Input(Bias) is [%d].",
            bias.dims()[0],
            right));
  }

  out->set_dims(x_dim);
  if (mean) {
    mean->set_dims({left});
  }
  if (variance) {
    variance->set_dims({left});
  }
  out->share_lod(x);
}

void LayerNormGradInferMeta(const MetaTensor& x,
                            const MetaTensor& y,
                            const MetaTensor& z,
                            MetaTensor* dx,
                            MetaTensor* dy,
                            MetaTensor* dz) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy && y) {
    dy->share_meta(y);
  }
  if (dz && z) {
    dz->share_meta(z);
  }
}

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto w_dims = weight.dims();
  DDim out_dims;
  out_dims = funcs::GetOutputDims(x_dims, y_dims);
  if (w_dims.size() > 1 || w_dims[0] != 1) {
    out_dims = funcs::GetOutputDims(out_dims, w_dims);
  }
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void LinspaceRawInferMeta(const MetaTensor& start,
                          const MetaTensor& stop,
                          const MetaTensor& number,
                          MetaTensor* out) {
  PADDLE_ENFORCE_EQ(
      phi::product(start.dims()),
      1,
      phi::errors::InvalidArgument("The size of Input(start) should be 1,"
                                   "but got %d.",
                                   phi::product(start.dims())));

  PADDLE_ENFORCE_EQ(
      phi::product(stop.dims()),
      1,
      phi::errors::InvalidArgument("The size of Input(stop) should be 1,"
                                   "but got %d.",
                                   phi::product(stop.dims())));

  PADDLE_ENFORCE_EQ(
      phi::product(number.dims()),
      1,
      phi::errors::InvalidArgument("The size of Input(number) should be 1,"
                                   "but got %d.",
                                   phi::product(number.dims())));

  out->set_dims(phi::make_ddim({-1}));
  out->set_dtype(start.dtype());
}

void LinspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       DataType dtype,
                       MetaTensor* out) {
  LinspaceRawInferMeta(start, stop, number, out);
}

void MultiClassNMSInferMeta(const MetaTensor& bboxes,
                            const MetaTensor& scores,
                            const MetaTensor& rois_num,
                            float score_threshold,
                            int nms_top_k,
                            int keep_top_k,
                            float nms_threshold,
                            bool normalized,
                            float nms_eta,
                            int background_label,
                            MetaTensor* out,
                            MetaTensor* index,
                            MetaTensor* nms_rois_num,
                            MetaConfig config) {
  auto box_dims = bboxes.dims();
  auto score_dims = scores.dims();
  auto score_size = score_dims.size();

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        score_size == 2 || score_size == 3,
        true,
        errors::InvalidArgument("The rank of Input(Scores) must be 2 or 3"
                                ". But received rank = %d",
                                score_size));
    PADDLE_ENFORCE_EQ(
        box_dims.size(),
        3,
        errors::InvalidArgument("The rank of Input(BBoxes) must be 3"
                                ". But received rank = %d",
                                box_dims.size()));
    if (score_size == 3) {
      PADDLE_ENFORCE_EQ(box_dims[2] == 4 || box_dims[2] == 8 ||
                            box_dims[2] == 16 || box_dims[2] == 24 ||
                            box_dims[2] == 32,
                        true,
                        errors::InvalidArgument(
                            "The last dimension of Input"
                            "(BBoxes) must be 4 or 8, "
                            "represents the layout of coordinate "
                            "[xmin, ymin, xmax, ymax] or "
                            "4 points: [x1, y1, x2, y2, x3, y3, x4, y4] or "
                            "8 points: [xi, yi] i= 1,2,...,8 or "
                            "12 points: [xi, yi] i= 1,2,...,12 or "
                            "16 points: [xi, yi] i= 1,2,...,16"));
      PADDLE_ENFORCE_EQ(
          box_dims[1],
          score_dims[2],
          errors::InvalidArgument(
              "The 2nd dimension of Input(BBoxes) must be equal to "
              "last dimension of Input(Scores), which represents the "
              "predicted bboxes."
              "But received box_dims[1](%s) != socre_dims[2](%s)",
              box_dims[1],
              score_dims[2]));
    } else {
      PADDLE_ENFORCE_EQ(box_dims[2],
                        4,
                        errors::InvalidArgument(
                            "The last dimension of Input"
                            "(BBoxes) must be 4. But received dimension = %d",
                            box_dims[2]));
      PADDLE_ENFORCE_EQ(
          box_dims[1],
          score_dims[1],
          errors::InvalidArgument(
              "The 2nd dimension of Input"
              "(BBoxes) must be equal to the 2nd dimension of Input(Scores). "
              "But received box dimension = %d, score dimension = %d",
              box_dims[1],
              score_dims[1]));
    }
  }
  PADDLE_ENFORCE_NE(out,
                    nullptr,
                    errors::InvalidArgument(
                        "The out in MultiClassNMSInferMeta can't be nullptr."));
  PADDLE_ENFORCE_NE(
      index,
      nullptr,
      errors::InvalidArgument(
          "The index in MultiClassNMSInferMeta can't be nullptr."));
  // Here the box_dims[0] is not the real dimension of output.
  // It will be rewritten in the computing kernel.

  out->set_dims(phi::make_ddim({-1, box_dims[2] + 2}));
  out->set_dtype(bboxes.dtype());
  index->set_dims(phi::make_ddim({-1, 1}));
  index->set_dtype(DataType::INT32);
  nms_rois_num->set_dims(phi::make_ddim({-1}));
  nms_rois_num->set_dtype(DataType::INT32);
}

void NllLossRawInferMeta(const MetaTensor& input,
                         const MetaTensor& label,
                         const MetaTensor& weight,
                         int64_t ignore_index,
                         const std::string& reduction,
                         MetaTensor* out,
                         MetaTensor* total_weight,
                         MetaConfig config) {
  auto x_dims = input.dims();
  auto label_dims = label.dims();
  PADDLE_ENFORCE_EQ(x_dims.size() == 2 || x_dims.size() == 4,
                    true,
                    phi::errors::InvalidArgument(
                        "The tensor rank of Input(X) must be 2 or 4."));
  bool contain_unknown_dim =
      phi::contain_unknown_dim(x_dims) || phi::contain_unknown_dim(label_dims);
  bool check = config.is_runtime || !contain_unknown_dim;
  if (check) {
    PADDLE_ENFORCE_EQ(
        x_dims[0],
        label_dims[0],
        phi::errors::InvalidArgument(
            "ShapeError: Expected input batch_size to match label batch_size,"
            "But received: the Input(x) batch_size is [%s], the Input(label) "
            " batch_size is [%s].",
            x_dims[0],
            label_dims[0]));
    if (weight) {
      auto w_dims = weight.dims();
      PADDLE_ENFORCE_EQ(
          w_dims.size(),
          1,
          phi::errors::InvalidArgument("Input(Weight) should be a 1D tensor."));
      PADDLE_ENFORCE_EQ(
          x_dims[1],
          w_dims[0],
          phi::errors::InvalidArgument(
              "Expected input tensor Weight's size should equal "
              "to the first dimension of the input tensor X. But received "
              "Weight's "
              "size is %d, the first dimension of input X is %d",
              w_dims[0],
              x_dims[1]));
    }
  }
  if (x_dims.size() == 2) {
    if (reduction == "none") {
      out->set_dims({x_dims[0]});
    } else {
      out->set_dims({1});
    }
  } else if (x_dims.size() == 4) {
    PADDLE_ENFORCE_EQ(label_dims.size(),
                      3,
                      phi::errors::InvalidArgument(
                          "Expected Input(Lable) dimensions=3, received %d.",
                          label_dims.size()));
    auto input0 = x_dims[0];
    auto input2 = x_dims[2];
    auto input3 = x_dims[3];
    auto label0 = label_dims[0];
    auto label1 = label_dims[1];
    auto label2 = label_dims[2];
    PADDLE_ENFORCE_EQ(
        input0 == label0 && input2 == label1 && input3 == label2,
        true,
        phi::errors::InvalidArgument("Input(X) tensor shape should "
                                     "match to Input(Label) tensor "
                                     "shape."));
    if (reduction == "none") {
      out->set_dims({x_dims[0], x_dims[2], x_dims[3]});
    } else {
      out->set_dims({1});
    }
  }
  total_weight->set_dims({1});
  out->set_dtype(input.dtype());
  total_weight->set_dtype(input.dtype());
}

void PutAlongAxisInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& value,
                           int axis,
                           const std::string& reduce,
                           MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void RoiAlignInferMeta(const MetaTensor& x,
                       const MetaTensor& boxes,
                       const MetaTensor& boxes_num,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       int sampling_ratio,
                       bool aligned,
                       MetaTensor* out,
                       MetaConfig config) {
  auto input_dims = x.dims();
  auto boxes_dims = boxes.dims();

  if (boxes_num) {
    auto boxes_num_dims = boxes_num.dims();
    PADDLE_ENFORCE_EQ(
        boxes_num_dims.size(),
        1,
        phi::errors::InvalidArgument("The size of boxes_num should be 1"
                                     ", but received size = %d",
                                     boxes_num_dims.size()));
  }
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "The format of Input(x) in"
                        "RoiAlignOp is NCHW. And the rank of input must be 4. "
                        "But received rank = %d",
                        input_dims.size()));
  PADDLE_ENFORCE_EQ(boxes_dims.size(),
                    2,
                    phi::errors::InvalidArgument("The rank of Input(boxes) "
                                                 "in RoiAlignOp should be 2. "
                                                 "But the rank of boxes is %d",
                                                 boxes_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(boxes_dims[1],
                      4,
                      phi::errors::InvalidArgument(
                          "The second dimension "
                          "of Input(boxes) should be 4. But received the "
                          "dimension = %d",
                          boxes_dims[1]));
  }

  PADDLE_ENFORCE_GT(pooled_height,
                    0,
                    phi::errors::InvalidArgument(
                        "The 'pooled_height' attribute in RoiAlignOp is "
                        "invalid. The height must be greater than 0. But "
                        "received 'pooled_height' = %d",
                        pooled_height));
  PADDLE_ENFORCE_GT(pooled_width,
                    0,
                    phi::errors::InvalidArgument(
                        "The 'pooled_width' attribute in RoiAlignOp is "
                        "invalid. The width must be greater than 0. But "
                        "received 'pooled_width' = %d",
                        pooled_width));
  PADDLE_ENFORCE_GT(spatial_scale,
                    0.0f,
                    phi::errors::InvalidArgument(
                        "The 'spatial_scale' attribute in RoiAlignOp is "
                        "invalid. The scale must be greater than 0. But "
                        "received 'spatial_scale' = %f",
                        spatial_scale));

  auto out_dims = input_dims;
  out_dims[0] = boxes_dims[0];
  out_dims[1] = input_dims[1];
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void RoiPoolInferMeta(const MetaTensor& x,
                      const MetaTensor& boxes,
                      const MetaTensor& boxes_num,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      MetaTensor* out,
                      MetaTensor* arg_max) {
  auto input_dims = x.dims();
  auto boxes_dims = boxes.dims();

  if (boxes_num) {
    auto boxes_num_dims = boxes_num.dims();
    PADDLE_ENFORCE_EQ(
        boxes_num_dims.size(),
        1,
        phi::errors::InvalidArgument("The second dimension of boxes_num should "
                                     "be 1, but received dimension is %d",
                                     boxes_num_dims.size()));
  }
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "The input data should be a four-dimensional "
                        "tensor with [N,C,H,W], but received input data with "
                        " %d dimension",
                        input_dims.size()));
  PADDLE_ENFORCE_EQ(
      boxes_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "boxes should be a 2-D LoDTensor with shape (num_boxes, 4)"
          "given as [[x1, y1, x2, y2], ...], but received boxes is "
          "%d-dimensional LoDTensor",
          boxes_dims.size()));
  PADDLE_ENFORCE_EQ(
      boxes_dims[1],
      4,
      phi::errors::InvalidArgument(
          "boxes should be a 2-D LoDTensor with shape (num_boxes, 4)"
          "given as [[x1, y1, x2, y2], ...]. But the second dimension of  "
          "the received data is %d",
          boxes_dims[1]));

  PADDLE_ENFORCE_GT(
      pooled_height,
      0,
      phi::errors::OutOfRange("The pooled output height must be greater than 0"
                              "but received height is %d",
                              pooled_height));
  PADDLE_ENFORCE_GT(
      pooled_width,
      0,
      phi::errors::OutOfRange("The pooled output width must be greater than 0"
                              "but received width is %d",
                              pooled_width));
  PADDLE_ENFORCE_GT(
      spatial_scale,
      0.0f,
      phi::errors::OutOfRange("The spatial scale must be greater than 0, "
                              "but received spatial scale is %f",
                              spatial_scale));

  auto out_dims = input_dims;
  out_dims[0] = boxes_dims[0];
  out_dims[1] = input_dims[1];
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  arg_max->set_dims(out_dims);
  arg_max->set_dtype(DataType::INT64);
}

void ScatterInferMeta(const MetaTensor& x,
                      const MetaTensor& index,
                      const MetaTensor& updates,
                      bool overwrite,
                      MetaTensor* out) {
  const auto& updates_dims = updates.dims();
  const auto& ref_dims = x.dims();
  const auto& index_dims = index.dims();

  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of the index should be 1 when the "
                          "index is a 2D tensor, but we get %d.",
                          index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size(),
        1,
        phi::errors::InvalidArgument("The index should be a 1D tensor when the "
                                     "index is not a 2D tensor, but we get %d.",
                                     index_dims.size()));
  }
  PADDLE_ENFORCE_EQ(
      ref_dims.size(),
      updates_dims.size(),
      phi::errors::InvalidArgument(
          "Input(X) and Input(Updates) should have the same shape size, "
          "but received the size of Input(x)'s shape is %d, the size of "
          "Input(Updates)'s shape is %d.",
          ref_dims.size(),
          updates_dims.size()));
  PADDLE_ENFORCE_EQ(
      updates_dims[0],
      index_dims[0],
      phi::errors::InvalidArgument(
          "Input(Updates) and Input(Ids) should have same batch-size, but"
          " received Input(Updates)'s batch-size is %d, Input(Ids)'s "
          "batch-size is %d.",
          updates_dims[0],
          index_dims[0]));
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ScatterNdAddInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& updates,
                           MetaTensor* out) {
  const auto& ref_dims = x.dims();
  auto ref_dims_size = ref_dims.size();
  const auto& index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  const auto& updates_dims = updates.dims();
  auto updates_dims_size = updates_dims.size();

  PADDLE_ENFORCE_LE(
      index_dims[index_dims_size - 1],
      ref_dims_size,
      phi::errors::InvalidArgument(
          "The last dimension of Input(Index)'s shape should be no greater "
          "than the rank of Input(X), but received the last dimension of "
          "Input(Index)'s shape is %d, the rank of Input(X) is %d.",
          index_dims[index_dims_size - 1],
          ref_dims_size));
  PADDLE_ENFORCE_GE(index_dims_size,
                    2UL,
                    phi::errors::InvalidArgument(
                        "The rank of Input(Index) should be greater than 1, "
                        "but received the rank of Input(Index) is %d.",
                        index_dims_size));

  // update.shape = index.shape[:-1] + output.shape[index.shape[-1]:]
  std::vector<int64_t> r_updates_dims;
  for (int64_t i = 0; i < index_dims_size - 1; ++i) {
    r_updates_dims.emplace_back(index_dims[i]);
  }
  for (int64_t i = index_dims[index_dims_size - 1]; i < ref_dims_size; ++i) {
    r_updates_dims.emplace_back(ref_dims[i]);
  }

  PADDLE_ENFORCE_EQ(
      r_updates_dims.size(),
      updates_dims_size,
      phi::errors::InvalidArgument(
          "Updates has wrong shape. The shape of Updates and Input(Updates) "
          "should be same, but received the shape of Updates is %d, "
          "the shape of Input(Updates) is %d.",
          r_updates_dims.size(),
          updates_dims_size));

  for (int64_t i = 0; i < updates_dims_size; ++i) {
    PADDLE_ENFORCE_EQ(
        r_updates_dims[i],
        updates_dims[i],
        phi::errors::InvalidArgument(
            "Updates has wrong shape. The dimensions of Updates and "
            "Input(Updates) should match, but received Updates's"
            "%d-th dimension is %d, Input(Updates)'s %d-th "
            "dimension is %d.",
            i,
            r_updates_dims[i],
            i,
            updates_dims[i]));
  }
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void SendURecvInferMeta(const MetaTensor& x,
                        const MetaTensor& src_index,
                        const MetaTensor& dst_index,
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

  auto dims = x.dims();
  std::vector<int64_t> dims_ = phi::vectorize(dims);
  dims_[0] = -1;
  out->set_dims(phi::make_ddim(dims_));
  out->set_dtype(x.dtype());

  if (reduce_op == "MEAN") {
    dst_count->set_dims({-1});
    dst_count->set_dtype(DataType::INT32);
  }
}

void SpectralNormInferMeta(const MetaTensor& weight,
                           const MetaTensor& u,
                           const MetaTensor& v,
                           int dim,
                           int power_iters,
                           float eps,
                           MetaTensor* out,
                           MetaConfig config) {
  auto dim_weight = weight.dims();
  auto rank_weight = dim_weight.size();
  PADDLE_ENFORCE_GE(rank_weight,
                    2,
                    errors::InvalidArgument(
                        "The rank of Input(Weights) should be greater equal "
                        "than 2, but received Weight rank(%d)",
                        rank_weight));
  PADDLE_ENFORCE_LE(
      rank_weight,
      5,
      errors::InvalidArgument("The rank of Input(Weights) should be less equal "
                              "than 5, but received Weight rank(%d)",
                              rank_weight));

  auto dim_valid = dim == 0 || dim == 1;
  PADDLE_ENFORCE_EQ(dim_valid,
                    true,
                    errors::InvalidArgument(
                        "Attr(dim) can only be 0 or 1, but received %d", dim));
  PADDLE_ENFORCE_GE(
      power_iters,
      0,
      errors::InvalidArgument(
          "Attr(power_iters) should be greater equal then 0, but received %d",
          power_iters));

  int h = dim_weight[dim];
  int w = 1;
  for (int i = 0; i < rank_weight; i++) {
    if (i != dim) {
      w *= dim_weight[i];
    }
  }
  auto dim_u = u.dims();
  auto dim_v = v.dims();

  if (config.is_runtime || (dim_u[0] > 0 && h > 0)) {
    PADDLE_ENFORCE_EQ(dim_u[0],
                      h,
                      errors::InvalidArgument(
                          "Input(U) dimension[0] should be equal to "
                          "Input(Weight) dimension[Attr(dim)], but received "
                          "U dimension[0](%d) != Weight dimension[%d](%d)",
                          dim_u[0],
                          dim,
                          h));
  }

  if (config.is_runtime || (dim_v[0] > 0 && w > 0)) {
    PADDLE_ENFORCE_EQ(
        dim_v[0],
        w,
        errors::InvalidArgument(
            "Input(V) dimension[0] should be equal to the product of "
            "Input(Weight) dimension except dimension[Attr(dim)], but "
            "received V dimension[0](%d) != product of Input(Weight) "
            "dimension(%d)",
            dim_v[0],
            w));
  }

  if (out) {
    out->set_dims(dim_weight);
    out->set_dtype(weight.dtype());
    out->share_lod(weight);
  }
}

void ViterbiDecodeInferMeta(const MetaTensor& input,
                            const MetaTensor& transition,
                            const MetaTensor& length,
                            bool include_bos_eos_tag,
                            MetaTensor* scores,
                            MetaTensor* path,
                            MetaConfig config) {
  auto in_dims = input.dims();
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The rank of Input in ViterbiDecode  must be 3. But "
                        "received Input's rank is %d.",
                        in_dims.size()));
  auto length_dims = length.dims();
  PADDLE_ENFORCE_EQ(length_dims.size(),
                    1,
                    phi::errors::InvalidArgument(
                        "The rank of Length in ViterbiDecode must be 1. But "
                        "received Length's rank is %d.",
                        length_dims.size()));
  auto transition_dims = transition.dims();
  PADDLE_ENFORCE_EQ(
      transition_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The rank of Transition in ViterbiDecode must be 2. But "
          "received Transition's rank is %d.",
          transition_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        in_dims[0],
        length_dims[0],
        phi::errors::InvalidArgument(
            "The batch size of Input and Length should be equal."));
    PADDLE_ENFORCE_EQ(in_dims[2],
                      transition_dims[0],
                      phi::errors::InvalidArgument(
                          "The number of tags of Input (%d) and Transition "
                          "(%d) should be equal.",
                          transition_dims[0],
                          in_dims[2]));
  }
  scores->set_dims(length_dims);
  scores->set_dtype(length.dtype());
}

}  // namespace phi
